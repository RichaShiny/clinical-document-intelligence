"""Frozen biomedical Transformer encoder plus learned logistic head.

Encoder parameters are NOT fine-tuned. Embeddings are cached with row order and revision.
"""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from .train import load_rows,partitions,score_candidate,finalize_artifacts,write_json
from .data import digest_file

MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
REVISION='e1354b7a3a09615f6aba48dfad4b7a613eef7062'
MAX_LENGTH=128


def embed(root,rows):
    torch.set_num_threads(4)
    cache=root/'runtime/biomed_embeddings.npz'
    fingerprint=digest_file(root/'runtime/prepared.json')+REVISION+str(MAX_LENGTH)
    if cache.exists():
        saved=np.load(cache)
        if str(saved['fingerprint'])==fingerprint:return saved['embeddings'],float(saved['seconds'])
    location=root/'runtime/biomedbert'
    tokenizer=AutoTokenizer.from_pretrained(str(location),local_files_only=True)
    model=AutoModel.from_pretrained(str(location),local_files_only=True).eval()
    order=sorted(range(len(rows)),key=lambda i:len(rows[i]['text']))
    output=np.zeros((len(rows),model.config.hidden_size),dtype=np.float32)
    start=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(order),24):
            idx=order[offset:offset+24]
            inputs=tokenizer([rows[i]['text'] for i in idx],return_tensors='pt',padding=True,truncation=True,max_length=MAX_LENGTH)
            hidden=model(**inputs).last_hidden_state
            mask=inputs['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1)
            output[idx]=pooled.numpy()
            if offset%480==0:print(f'BiomedBERT embeddings {offset}/{len(rows)} ({time.perf_counter()-start:.1f}s)',flush=True)
    seconds=time.perf_counter()-start
    np.savez_compressed(cache,embeddings=output,fingerprint=fingerprint,seconds=seconds)
    return output,seconds


def train(root):
    rows=load_rows(root);parts=partitions(rows);embedding,seconds=embed(root,rows)
    indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in parts}
    x={s:embedding[idx] for s,idx in indices.items()};y={s:np.array([r['label'] for r in parts[s]]) for s in parts}
    scale=StandardScaler().fit(x['train']);x={s:scale.transform(v) for s,v in x.items()};best=None;trials=[]
    for c in [.01,.1,1.]:
        model=LogisticRegression(C=c,max_iter=1000,random_state=42).fit(x['train'],y['train'])
        ap=average_precision_score(y['tune'],model.predict_proba(x['tune'])[:,1]);trials.append(dict(C=c,tune_auprc=float(ap)))
        if best is None or ap>best[0]:best=(ap,model,c)
    predictions={s:best[1].predict_proba(x[s])[:,1] for s in parts if s!='train'}
    rep,cal,probs,threshold=score_candidate('biomedbert_frozen',predictions,parts,dict(framework='Hugging Face Transformers + PyTorch + scikit-learn',
        model_id=MODEL,revision=REVISION,encoder_finetuned=False,pooling='attention-masked mean',max_length=MAX_LENGTH,
        embedding_seconds=seconds,C=best[2],tuning_trials=trials))
    joblib.dump(dict(scaler=scale,model=best[1],calibrator=cal,threshold=threshold),root/'runtime/artifacts/biomedbert_head.joblib')
    report=json.loads((root/'reports/metrics.json').read_text());report['models']['biomedbert_frozen']=rep
    report['selected_model']=max(report['models'],key=lambda n:report['models'][n]['selection_auprc'])
    report['selection_rule']='Highest tuning-split AUPRC among all evaluated candidates; review threshold fixed on independent policy split.'
    write_json(root/'reports/metrics.json',report)
    local=json.loads((root/'runtime/test_predictions.json').read_text())
    for i,row in enumerate(local):row['predictions']['biomedbert_frozen']=float(probs['test'][i])
    write_json(root/'runtime/test_predictions.json',local)
    from . import entities
    crf=joblib.load(root/'runtime/artifacts/entities_crf.joblib')
    chosen=report['selected_model']
    report['gated_entity_extraction']=entities.evaluate(crf,parts['test'],{r['id']:bool(r['predictions'][chosen]>=report['models'][chosen]['test']['threshold']) for r in local})
    write_json(root/'reports/metrics.json',report)
    finalize_artifacts(root,report)
    print('BIOMEDBERT test AP',rep['test']['auprc'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());train(p.parse_args().root)

if __name__=='__main__':main()
