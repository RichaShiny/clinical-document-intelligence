"""Partially fine-tuned BiomedBERT BIO tagger for ADE-associated source spans.

Only the last two encoder blocks and classification head are updated. Selection uses
exact-span tuning F1; the existing document-separated test set is used for comparison.
"""
import argparse
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from . import entities
from .biomed import MODEL, REVISION
from .data import digest_file
from .train import load_rows, partitions, write_json, finalize_artifacts

LABELS = ['O', 'B-DRUG', 'I-DRUG', 'B-EFFECT', 'I-EFFECT']
MAX_LENGTH = 128


def align_labels(offsets, annotations):
    """Align complete character spans; ignore specials and partial/unrepresentable spans."""
    labels = [0 if end > start else -100 for start, end in offsets]
    ignored = 0
    for entity in annotations:
        overlapping = [i for i, (a, b) in enumerate(offsets)
                       if b > a and a < entity['end'] and b > entity['start']]
        complete = (overlapping and offsets[overlapping[0]][0] == entity['start']
                    and offsets[overlapping[-1]][1] == entity['end'])
        if not complete or any(labels[i] != 0 for i in overlapping):
            ignored += 1
            # Do not overwrite an already represented overlapping entity.
            for i in overlapping:
                if labels[i] == 0:
                    labels[i] = -100
            continue
        begin = LABELS.index('B-' + entity['kind'])
        inside = LABELS.index('I-' + entity['kind'])
        for j, i in enumerate(overlapping):
            labels[i] = begin if j == 0 else inside
    return labels, ignored


def decode_offsets(text, offsets, tags):
    if len(offsets) != len(tags):
        raise ValueError('Offset/tag length mismatch')
    spans = []
    current = None
    for (start, end), tag in zip(offsets, tags):
        if end <= start:
            continue
        if tag == 'O':
            current = None
            continue
        prefix, kind = tag.split('-', 1)
        if prefix == 'B' or current is None or current['kind'] != kind:
            current = dict(kind=kind, start=int(start), end=int(end))
            spans.append(current)
        else:
            current['end'] = int(end)
    return [dict(e, text=text[e['start']:e['end']]) for e in spans]


def span_metrics(rows, predictions, gates=None):
    if len(rows) != len(predictions):
        raise ValueError('Prediction count must match evaluation rows')
    result = {}
    totals = np.zeros(3, dtype=int)
    for kind in ['DRUG', 'EFFECT']:
        counts = np.zeros(3, dtype=int)
        for row, predicted in zip(rows, predictions):
            gold = {(e['start'], e['end']) for e in row['entities'] if e['kind'] == kind}
            candidate = predicted if gates is None or gates.get(row['id'], False) else []
            found = {(e['start'], e['end']) for e in candidate if e['kind'] == kind}
            counts += [len(gold & found), len(found - gold), len(gold - found)]
        a, b, c = map(int, counts)
        result[kind] = dict(tp=a, fp=b, fn=c, f1=2*a/(2*a+b+c) if 2*a+b+c else 0.)
        totals += counts
    a, b, c = map(int, totals)
    return dict(tp=a, fp=b, fn=c, precision=a/(a+b) if a+b else 0.,
                recall=a/(a+c) if a+c else 0., micro_f1=2*a/(2*a+b+c) if 2*a+b+c else 0.,
                by_kind=result, scope='Exact ADE-associated type/start/end spans; all supplied sentences.')


def paired_span_delta(rows, baseline, candidate, samples=300):
    """Document-group paired bootstrap of candidate minus baseline exact-span F1."""
    if not (len(rows) == len(baseline) == len(candidate)):
        raise ValueError('Prediction count must match evaluation rows')
    def counts(row, predicted):
        gold = {(e['kind'],e['start'],e['end']) for e in row['entities']}
        found = {(e['kind'],e['start'],e['end']) for e in predicted}
        return np.array([len(gold & found),len(found-gold),len(gold-found)])
    def f1(totals):
        a,b,c = totals
        return float(2*a/(2*a+b+c)) if 2*a+b+c else 0.
    groups = sorted({r['group'] for r in rows})
    a = {g:np.zeros(3,dtype=int) for g in groups}
    b = {g:np.zeros(3,dtype=int) for g in groups}
    for row, first, second in zip(rows,baseline,candidate):
        a[row['group']] += counts(row,first)
        b[row['group']] += counts(row,second)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(samples):
        draw = rng.choice(groups,len(groups),replace=True)
        deltas.append(f1(sum((b[g] for g in draw),np.zeros(3,dtype=int))) -
                      f1(sum((a[g] for g in draw),np.zeros(3,dtype=int))))
    return dict(delta_f1=f1(sum(b.values()))-f1(sum(a.values())),
                ci95=np.quantile(deltas,[.025,.975]).tolist(),samples=samples,
                unit='Connected source-document groups; paired bootstrap; seed42')


def build_model(location):
    return AutoModelForTokenClassification.from_pretrained(
        str(location), local_files_only=True, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label:i for i,label in enumerate(LABELS)})


class TokenExtractor:
    def __init__(self, root, model=None, tokenizer=None):
        self.root = Path(root)
        location = self.root/'runtime/biomedbert'
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(str(location), local_files_only=True)
        self.model = model if model is not None else build_model(location)
        if model is None:
            delta = torch.load(self.root/'runtime/artifacts/entities_biomedbert_delta.pt',
                               map_location='cpu', weights_only=True)
            expected = {n for n,_ in self.model.named_parameters()
                        if n.startswith(('bert.encoder.layer.10.', 'bert.encoder.layer.11.', 'classifier.'))}
            if set(delta) != expected:
                raise ValueError('Invalid biomedical entity checkpoint parameter set')
            for name, tensor in delta.items():
                if tensor.shape != self.model.state_dict()[name].shape:
                    raise ValueError('Invalid biomedical entity checkpoint shape')
            self.model.load_state_dict(delta, strict=False)
        self.model.eval()

    def predict_batch(self, texts, batch_size=24):
        output = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start+batch_size]
                encoded = self.tokenizer(batch, return_tensors='pt', padding=True,
                                         truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True)
                offsets = encoded.pop('offset_mapping').tolist()
                labels = self.model(**encoded).logits.argmax(-1).tolist()
                for text, spans, ids in zip(batch, offsets, labels):
                    output.append(decode_offsets(text, spans, [LABELS[i] for i in ids]))
        return output


def prepare_examples(tokenizer, rows):
    encoded = tokenizer([r['text'] for r in rows], truncation=True,
                        max_length=MAX_LENGTH, return_offsets_mapping=True)
    examples = []
    ignored = 0
    for i, row in enumerate(rows):
        labels, count = align_labels(encoded['offset_mapping'][i], row['entities'])
        ignored += count
        item = {k: v[i] for k,v in encoded.items() if k != 'offset_mapping'}
        item['labels'] = labels
        examples.append(item)
    return examples, ignored


def collate(tokenizer, examples):
    batch = tokenizer.pad([{k:v for k,v in e.items() if k != 'labels'} for e in examples],
                          padding=True, return_tensors='pt')
    length = batch['input_ids'].shape[1]
    labels = [e['labels'] + [-100]*(length-len(e['labels'])) for e in examples]
    return batch, torch.tensor(labels, dtype=torch.long)


def train(root, epochs=3):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    root = Path(root)
    rows = load_rows(root)
    parts = partitions(rows)
    positives = [r for r in parts['train'] if r['label'] == 1]
    negatives = [r for r in parts['train'] if r['label'] == 0]
    training = positives + random.Random(42).sample(negatives, min(len(negatives), 2*len(positives)))
    tokenizer = AutoTokenizer.from_pretrained(str(root/'runtime/biomedbert'), local_files_only=True)
    examples, ignored = prepare_examples(tokenizer, training)
    model = build_model(root/'runtime/biomedbert')
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(('bert.encoder.layer.10.', 'bert.encoder.layer.11.', 'classifier.'))
    trainable = {name for name,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')], 'lr':2e-5},
        {'params':[p for n,p in model.named_parameters() if n.startswith('classifier.')], 'lr':5e-4}], weight_decay=.01)
    # Weights depend only on the training tokens; soften imbalance without discarding negatives.
    counts = np.bincount([v for e in examples for v in e['labels'] if v >= 0], minlength=len(LABELS))
    weights = np.sqrt(counts.max()/np.maximum(counts,1))
    weights = np.minimum(weights, 8.)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), ignore_index=-100)
    extractor = TokenExtractor(root, model=model, tokenizer=tokenizer)
    history = []
    best = None
    rng = np.random.default_rng(42)
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        model.bert.embeddings.eval()
        for layer in model.bert.encoder.layer[:10]:layer.eval()
        order = rng.permutation(len(examples))
        losses = []
        for step, offset in enumerate(range(0, len(order), 16)):
            indices = order[offset:offset+16]
            batch, labels = collate(tokenizer, [examples[i] for i in indices])
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits.reshape(-1,len(LABELS)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(f'NER epoch {epoch}/{epochs} batch {step} loss {np.mean(losses):.4f} elapsed {time.perf_counter()-start:.1f}s', flush=True)
        tune_predictions = extractor.predict_batch([r['text'] for r in parts['tune']])
        score = span_metrics(parts['tune'], tune_predictions)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),tune=score))
        if best is None or score['micro_f1'] > best[0]:
            best = (score['micro_f1'], epoch,
                    {n:p.detach().clone() for n,p in model.named_parameters() if n in trainable})
        print(f'NER epoch {epoch}: tuning exact-span F1 {score["micro_f1"]:.4f}',flush=True)
    model.load_state_dict(best[2], strict=False)
    crf = joblib.load(root/'runtime/artifacts/entities_crf.joblib')
    crf_tune = entities.evaluate(crf,parts['tune'])
    # Promotion decision is made before evaluating this candidate on test.
    promoted = best[0] > crf_tune['micro_f1']
    torch.save(best[2], root/'runtime/artifacts/entities_biomedbert_delta.pt')
    test_predictions = extractor.predict_batch([r['text'] for r in parts['test']])
    candidate_test = span_metrics(parts['test'],test_predictions)
    crf_test = entities.evaluate(crf,parts['test'])
    baseline_predictions = [entities.predict(crf,r['text']) for r in parts['test']]
    paired = paired_span_delta(parts['test'],baseline_predictions,test_predictions)
    local = json.loads((root/'runtime/test_predictions.json').read_text())
    metrics = json.loads((root/'reports/metrics.json').read_text())
    chosen = metrics['selected_model']
    scores = {r['id']:r['predictions'][chosen] for r in local}
    gates = {r['id']:scores[r['id']] >= metrics['models'][chosen]['test']['threshold'] for r in parts['test']}
    gated = span_metrics(parts['test'],test_predictions,gates)
    report = dict(model=MODEL,revision=REVISION,seed=42,epochs=epochs,best_epoch=best[1],
        trainable_encoder_blocks=[10,11],frozen_encoder_blocks=list(range(10)),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        max_length=MAX_LENGTH,train_sentences=len(training),train_ignored_entities=ignored,
        class_weights=weights.tolist(),history=history,training_and_evaluation_seconds=time.perf_counter()-start,
        data_sha256=digest_file(root/'runtime/prepared.json'),crf_tune=crf_tune,candidate_tune_f1=best[0],
        crf_test=crf_test,candidate_test=candidate_test,candidate_gated_test=gated,
        paired_test_delta=paired,training_configuration=dict(batch_size=16,encoder_learning_rate=2e-5,
            head_learning_rate=5e-4,weight_decay=.01,gradient_clip=1.,
            negative_to_positive_sentence_ratio=2,optimizer='AdamW'),
        selected_extractor='biomedbert_token' if promoted else 'crf',
        selection_rule='Promote only if tuning exact type/start/end micro F1 exceeds existing CRF; test not used for promotion.',
        limitation='Existing development benchmark; not an untouched external clinical validation. Sparse, ambiguous and overlapping annotations remain.')
    write_json(root/'reports/entity_experiment.json',report)
    metrics.setdefault('entity_baseline',metrics['entity_extraction'])
    metrics['entity_models'] = dict(crf=crf_test,biomedbert_token=candidate_test)
    metrics['selected_extractor'] = report['selected_extractor']
    if promoted:
        metrics['entity_extraction'] = dict(candidate_test,framework='Hugging Face Transformers / PyTorch',
            encoder_finetuned_blocks=[10,11],selection_tune_f1=best[0])
        metrics['gated_entity_extraction'] = gated
        predictions_by_id = {r['id']:p for r,p in zip(parts['test'],test_predictions)}
        for row in local:row['entities'] = predictions_by_id[row['id']]
        write_json(root/'runtime/test_predictions.json',local)
    write_json(root/'reports/metrics.json',metrics)
    finalize_artifacts(root,metrics)
    print(json.dumps(dict(selected=report['selected_extractor'],crf_test_f1=crf_test['micro_f1'],
                          candidate_test_f1=candidate_test['micro_f1'],gated_f1=gated['micro_f1']),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--epochs',type=int,default=3)
    args = parser.parse_args()
    if args.epochs < 1:parser.error('--epochs must be positive')
    train(args.root,args.epochs)

if __name__ == '__main__':main()
