"""Train/evaluate real-data candidates; test labels never drive selection or policy."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import torch
from .data import assert_disjoint,digest_file
from .metrics import calibrator,calibrated,choose_threshold,evaluate,cluster_bootstrap
from . import neural,entities


def load_rows(root):
    rows=json.loads((root/'runtime/prepared.json').read_text())
    assert_disjoint(rows)
    return rows


def partitions(rows):return {s:[r for r in rows if r['split']==s] for s in ['train','tune','calibration','policy','test']}


def write_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def score_candidate(name,predictions,parts,metadata):
    y={s:np.array([r['label'] for r in rows]) for s,rows in parts.items()}
    cal=calibrator(y['calibration'],predictions['calibration'])
    probs={s:calibrated(cal,p) for s,p in predictions.items()}
    threshold=choose_threshold(y['policy'],probs['policy'])
    report=dict(name=name,metadata=metadata,selection_auprc=float(average_precision_score(y['tune'],probs['tune'])),
        policy_target_recall=.90,policy=evaluate(y['policy'],probs['policy'],threshold),
        test=evaluate(y['test'],probs['test'],threshold),
        uncalibrated_test_brier=float(np.mean((y['test']-predictions['test'])**2)),
        test_ci95=cluster_bootstrap(y['test'],probs['test'],[r['group'] for r in parts['test']],threshold),
        slices={})
    for slice_name,mask in [('short_le_15_words',np.array([len(r['text'].split())<=15 for r in parts['test']])),
                            ('long_gt_15_words',np.array([len(r['text'].split())>15 for r in parts['test']]))]:
        if len(set(y['test'][mask]))==2:report['slices'][slice_name]=evaluate(y['test'][mask],probs['test'][mask],threshold)
    return report,cal,probs,threshold


def train(root):
    np.random.seed(42);random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    rows=load_rows(root);parts=partitions(rows)
    text={s:[r['text'] for r in v] for s,v in parts.items()}
    y={s:np.array([r['label'] for r in v]) for s,v in parts.items()}
    artifacts=root/'runtime/artifacts';artifacts.mkdir(exist_ok=True)
    report=dict(task='ADE literature sentence triage',seed=42,python=platform.python_version(),platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','scikit-learn','sklearn-crfsuite','pandas']},
        models={},selection_rule='Highest tuning-split AUPRC, chosen before examining test scores.',
        split_sizes={s:len(v) for s,v in parts.items()},deployment_status='Local research API; not clinically validated or deployed to healthcare systems.')
    probability_exports={}
    # Feature construction learns vocabulary and IDF only from train.
    vectorizer=FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=40000,sublinear_tf=True)),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=30000,sublinear_tf=True))])
    features={'train':vectorizer.fit_transform(text['train'])}
    for s in parts:
        if s!='train':features[s]=vectorizer.transform(text[s])
    best=None;trials=[];start=time.perf_counter()
    for c in [.25,1.,4.]:
        candidate=LogisticRegression(C=c,max_iter=500,solver='liblinear',random_state=42).fit(features['train'],y['train'])
        ap=average_precision_score(y['tune'],candidate.predict_proba(features['tune'])[:,1]);trials.append(dict(C=c,tune_auprc=float(ap)))
        if best is None or ap>best[0]:best=(ap,candidate,c)
    model=best[1]
    predictions={s:model.predict_proba(features[s])[:,1] for s in parts if s!='train'}
    rep,cal,probs,threshold=score_candidate('tfidf_logistic',predictions,parts,dict(C=best[2],tuning_trials=trials,
        framework='scikit-learn',train_seconds=time.perf_counter()-start,features='Word 1–2 grams and character 3–5 grams'))
    pipeline=Pipeline([('features',vectorizer),('classifier',model)])
    joblib.dump(dict(model=pipeline,calibrator=cal,threshold=threshold),artifacts/'tfidf_logistic.joblib')
    report['models']['tfidf_logistic']=rep;probability_exports['tfidf_logistic']=probs
    print('TFIDF',rep['test']['auprc'],flush=True)
    vocab=neural.vocabulary(text['train']);encoded={s:neural.encode(text[s],vocab) for s in parts}
    net=neural.DeepAverage(len(vocab)+1);optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
    loss_fn=torch.nn.BCEWithLogitsLoss();rng=np.random.default_rng(42);best_ap=-1;state=None;history=[];start=time.perf_counter()
    for epoch in range(1,21):
        net.train();losses=[];order=rng.permutation(len(y['train']))
        for offset in range(0,len(order),128):
            idx=order[offset:offset+128];batch=[encoded['train'][i] for i in idx]
            optimizer.zero_grad();logits=net(*neural.batch_tokens(batch));loss=loss_fn(logits,torch.tensor(y['train'][idx],dtype=torch.float32))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        tune=neural.predict(net,encoded['tune']);ap=float(average_precision_score(y['tune'],tune))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),tune_auprc=ap))
        if ap>best_ap:best_ap=ap;state=copy.deepcopy(net.state_dict());best_epoch=epoch
        print('DAN epoch',epoch,'tune AP',round(ap,4),flush=True)
    net.load_state_dict(state)
    predictions={s:neural.predict(net,encoded[s]) for s in parts if s!='train'}
    rep,cal,probs,threshold=score_candidate('pytorch_dan',predictions,parts,dict(framework='PyTorch',
        architecture='96-dim trainable EmbeddingBag → 64 ReLU → dropout .25 → binary logit',
        train_seconds=time.perf_counter()-start,best_epoch=best_epoch,history=history,vocabulary=len(vocab),max_tokens=256))
    torch.save(net.state_dict(),artifacts/'pytorch_dan.pt')
    write_json(artifacts/'vocabulary.json',vocab)
    joblib.dump(dict(calibrator=cal,threshold=threshold),artifacts/'pytorch_dan_policy.joblib')
    report['models']['pytorch_dan']=rep;probability_exports['pytorch_dan']=probs
    start=time.perf_counter();entity_trials=[];best_crf=None
    positive=[r for r in parts['train'] if r['label']==1]
    negative=[r for r in parts['train'] if r['label']==0]
    balanced=positive+random.Random(42).sample(negative,len(positive))
    for policy,training in [('all_sentences',parts['train']),('balanced_sentences',balanced)]:
        candidate,overlap=entities.train(training)
        tune=entities.evaluate(candidate,parts['tune'])['micro_f1']
        entity_trials.append(dict(policy=policy,n=len(training),tune_exact_micro_f1=tune))
        if best_crf is None or tune>best_crf[0]:best_crf=(tune,candidate,overlap,policy,len(training))
    _,crf,overlap,entity_policy,entity_n=best_crf
    joblib.dump(crf,artifacts/'entities_crf.joblib')
    report['entity_extraction']=dict(**entities.evaluate(crf,parts['test']),train_seconds=time.perf_counter()-start,
        overlapping_train_spans_excluded=overlap,framework='sklearn-crfsuite / CRFsuite',train_sentences=entity_n,
        selected_training_policy=entity_policy,tuning_trials=entity_trials,lexicon='Entity-token membership fitted on training annotations only')
    chosen=max(report['models'],key=lambda n:report['models'][n]['selection_auprc'])
    report['selected_model']=chosen
    report['gated_entity_extraction']=entities.evaluate(crf,parts['test'],{r['id']:bool(probability_exports[chosen]['test'][i]>=report['models'][chosen]['test']['threshold']) for i,r in enumerate(parts['test'])})
    write_json(root/'reports/metrics.json',report)
    # Derived outputs remain local; no copyrighted case-report sentences are committed.
    local=[]
    for i,row in enumerate(parts['test']):
        local.append(dict(id=row['id'],pmids=row['pmids'],group=row['group'],text=row['text'],label=row['label'],
            predictions={n:float(p['test'][i]) for n,p in probability_exports.items()},
            gold_entities=row['entities'],entities=entities.predict(crf,row['text'])))
    write_json(root/'runtime/test_predictions.json',local)
    finalize_artifacts(root,report)
    print(json.dumps({n:r['test']['auprc'] for n,r in report['models'].items()}),flush=True)


def finalize_artifacts(root,report):
    artifacts=root/'runtime/artifacts'
    manifest=dict(selected_model=report['selected_model'],selected_extractor=report.get('selected_extractor','crf'),dataset_sha256=digest_file(root/'runtime/prepared.json'),
        files={f.name:digest_file(f) for f in artifacts.iterdir() if f.is_file() and f.name!='manifest.json'},
        encoder_files={f.name:digest_file(f) for f in (root/'runtime/biomedbert').glob('*') if f.is_file()} if report['selected_model']=='biomedbert_frozen' or report.get('selected_extractor')=='biomedbert_token' else {},
        note='Load only locally generated trusted model artifacts. Digests detect corruption, not malicious replacement of both manifest and files.')
    write_json(artifacts/'manifest.json',manifest)
    write_json(root/'reports/artifact_manifest.json',manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());train(p.parse_args().root)

if __name__=='__main__':main()
