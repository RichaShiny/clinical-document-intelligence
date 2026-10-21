# Development-test error analysis

Run after training the sentence classifiers and promoted entity extractor:

```sh
python -m clinical_ade.error_analysis
```

The command reads private `runtime/test_predictions.json` and the selected model's policy threshold from `reports/metrics.json`. It writes only aggregate counts and rates to `reports/error_analysis.json`, with a SHA-256 digest of its prediction input. It does not perform inference, retune the policy, select a model, or publish case text, document identifiers, entity strings, or individual predictions. Regenerate the report whenever predictions or model selection change; an input digest identifies the prediction file but does not certify its provenance.

The checked-in report uses the same previously inspected 4,172 development-test sentences as the model report. There are 747 detected positive sentences, 106 missed positive sentences, and 457 false positive review candidates. Recall is 87.6%, below the policy's 90% target, while 28.9% of sentences are sent to review.

| Sentence characters | Sentences | Positive sentences | Recall | Precision | Review fraction |
|---|---:|---:|---:|---:|---:|
| 0–160 | 3,157 | 620 | 88.4% | 64.8% | 26.8% |
| 161–320 | 979 | 227 | 85.0% | 55.6% | 35.4% |
| 321+ | 36 | 6 | 100.0% | 54.5% | 30.6% |

These are descriptive slices without confidence intervals. The longest group contains only six positives; its recall is too weakly supported to establish reliable long-document performance. Character length is not WordPiece length, so these results do not establish truncation as the cause of errors. No demographic attributes are available for a fairness analysis.

Exact drug-span F1 after sentence gating is 0.670; effect-span F1 is 0.589. In the 161–320 character slice these fall to 0.588 and 0.510. This identifies a useful investigation target: manually adjudicate missed effects and span boundaries, including token truncation, before choosing the next modeling change. Sparse ADE annotations mean an unmatched predicted span is not necessarily a clinically false assertion. Gated metrics combine sentence triage misses with extraction errors; they do not isolate tagger quality.

Tests cover character-boundary assignment, threshold equality, suppression of spans on unflagged sentences, confusion counts, absent denominators, invalid inputs, and exclusion of source text from report output. This is internal diagnostic evidence, not independent clinical validation or a deployment approval.
