"""Re-evaluate saved LLM outputs after deterministic source-span normalization.

No model requests. Preserve strict-v1 metrics and show the post-hoc nature of this experiment.
"""
import argparse
import copy
import json
from pathlib import Path
from .llm import validate
from .evaluate import evaluate, paired_delta, key
from .extract import summarize, extract


def replay(root):
    raw=json.loads((root/'reports/llm_raw.json').read_text())
    payload=json.loads((root/'dashboard/results.json').read_text())
    rows={r['id']:r for r in payload['notes']}
    predictions={}
    name='ollama:'+raw['model']
    rejected=0
    for item in raw['entities']:
        row=rows[item['id']]
        accepted,invalid=validate({'entities':item['entities']},row['text'])
        predictions[row['id']]=accepted
        rejected+=len(invalid)
        g,p={key(e) for e in row['gold']},{key(e) for e in accepted}
        row['predictions'][name]=dict(entities=accepted,summary=summarize(row['text'],accepted),
            false_positives=[list(e) for e in sorted(p-g)],false_negatives=[list(e) for e in sorted(g-p)],
            raw_entities=item['entities'],rejected=invalid,
            prompt_tokens=item.get('prompt_tokens'),output_tokens=item.get('output_tokens'))
    subset=[r for r in payload['notes'] if r['id'] in predictions]
    system=payload['report']['systems'].setdefault(name,copy.deepcopy(raw['original_metrics']))
    payload['report']['llm_status']='executed (saved-response normalization replay)'
    system.update(splits=evaluate(subset,predictions),rejected_entities=rejected,
        validation_policy='normalized-v2: unique alias inside exact source quote; source-derived follow-up text',
        latency_scope='Original inference + strict-v1 validation; offline normalization replay excluded')
    baseline={r['id']:extract(r['text']) for r in subset}
    payload['report']['llm_comparison']=dict(baseline_on_same_subset=evaluate(subset,baseline),
        paired_test_delta=paired_delta(subset,baseline,predictions),
        experiment='Post-hoc alignment improvement on the same saved responses. Not an untouched holdout result.',
        strict_v1_metrics=raw['original_metrics'])
    encoded=json.dumps(payload,ensure_ascii=False)
    (root/'dashboard/results.json').write_text(encoded+'\n')
    (root/'dashboard/results.js').write_text('window.CLINICAL_DATA = '+encoded.replace('<','\\u003c')+';\n')
    (root/'reports/metrics.json').write_text(json.dumps(payload['report'],indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({split:{'n':v['n'],'f1':v['f1']} for split,v in system['splits'].items()},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    replay(parser.parse_args().root)

if __name__=='__main__':main()
