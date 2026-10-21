"""Aggregate-only diagnostic report from saved development-test predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(rows, model, threshold):
    """Keep text/identifiers private; undefined rates are null, never zero."""
    if not rows:
        raise ValueError('No predictions to analyze')
    buckets = {'all': rows, 'chars_0_160': [], 'chars_161_320': [], 'chars_321_plus': []}
    for row in rows:
        score = row['predictions'][model]
        if row['label'] not in (0, 1) or not 0 <= score <= 1:
            raise ValueError('Invalid label or probability')
        size = len(row['text'])
        buckets['chars_0_160' if size <= 160 else 'chars_161_320' if size <= 320 else 'chars_321_plus'].append(row)
    result = {}
    for name, subset in buckets.items():
        tp = fp = tn = fn = 0
        spans = {kind: dict(tp=0, fp=0, fn=0) for kind in ('DRUG', 'EFFECT')}
        for row in subset:
            flagged = row['predictions'][model] >= threshold
            positive = row['label'] == 1
            tp += int(flagged and positive)
            fp += int(flagged and not positive)
            tn += int(not flagged and not positive)
            fn += int(not flagged and positive)
            for kind, counts in spans.items():
                gold = {(e['start'], e['end']) for e in row['gold_entities'] if e['kind'] == kind}
                pred = {(e['start'], e['end']) for e in row['entities'] if e['kind'] == kind} if flagged else set()
                counts['tp'] += len(gold & pred)
                counts['fp'] += len(pred - gold)
                counts['fn'] += len(gold - pred)
        for counts in spans.values():
            denom = 2 * counts['tp'] + counts['fp'] + counts['fn']
            counts['f1'] = 2 * counts['tp'] / denom if denom else None
        result[name] = dict(n=len(subset), tp=tp, fp=fp, tn=tn, fn=fn,
                            recall=tp / (tp + fn) if tp + fn else None,
                            precision=tp / (tp + fp) if tp + fp else None,
                            review_fraction=(tp + fp) / len(subset) if subset else None,
                            gated_exact_spans=spans)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=Path('runtime/test_predictions.json'))
    parser.add_argument('--metrics', type=Path, default=Path('reports/metrics.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/error_analysis.json'))
    args = parser.parse_args()
    raw = args.predictions.read_bytes()
    metrics = json.loads(args.metrics.read_text())
    model = metrics['selected_model']
    threshold = metrics['models'][model]['policy']['threshold']
    report = dict(model=model, extractor=metrics.get('selected_extractor', 'crf'), threshold=threshold,
                  predictions_sha256=hashlib.sha256(raw).hexdigest(),
                  scope='Previously inspected development test set; descriptive diagnostics, not external validation. '
                        'Character length is not token length or a demographic subgroup. Sparse gold annotations limit span interpretation. '
                        'No thresholds or models are selected by these slices.',
                  slices=analyze(json.loads(raw), model, threshold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
