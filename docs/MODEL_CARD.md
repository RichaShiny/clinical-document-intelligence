# System/model card

**Intended use:** local research and portfolio demonstration of literature safety-review prioritization. Users are analysts reviewing published case reports. The tool retains evidence and manual reviewer judgment.

**Excluded uses:** patient diagnosis or treatment, medication discontinuation, clinical triage, autonomous safety-case filing, regulatory conclusions, causal attribution, and claims of verified adverse-event incidence. There has been no clinical validation or deployment in a healthcare organization.

## Training and inference

The dataset, five-way group split, measured model parameters and scores are documented in the README and reports. The default candidate is selected by tune AUPRC. For sentence classification, BiomedBERT is a frozen biomedical text encoder with a learned standardized logistic head; the encoder itself was not trained or fine-tuned here. Other trained components include lexical logistic regression, a scratch PyTorch Deep Averaging Network, calibrators, a CRF, and an optional biomedical token classifier. The token classifier partially fine-tunes the final two encoder blocks and BIO head; its encoder weights are separate from the frozen sentence-classification encoder.

The API groups submitted text into heuristic sentences, estimates an ADE-related probability, applies the learned review threshold and runs the selected entity extractor on flagged sentences. Entities and summary evidence remain verbatim source spans. Output is reviewable evidence, not a generated clinical assessment. A “lower priority” sentence is not declared safe or ADE-free.

## Actual limitations

Ranking is stronger than exact entity extraction. The latter's lower F1 is visible in both ungated and end-to-end metrics. Thresholds fitted to 90% policy recall can underperform that target on test; this happens in the measured biomedical candidate. Corpus prevalence and writing style differ from general clinical documents and routine publications. Probability calibration is distribution-dependent.

Annotations are sparse and selected; not every drug or condition mention is a target. Overlapping and ambiguous mentions reduce evaluable coverage. The model does not distinguish event severity, chronology or causal certainty reliably, and it does not link all predicted drugs to effects. Pretraining text may overlap source literature. Human review remains necessary even when source spans are correct.

## Promotion requirements

A real deployment needs authorized data/model use, an independent representative evaluation with clinician adjudication, risk-based acceptance criteria, workload/usability studies, access/security review, production monitoring, and a documented release/rollback owner. Those requirements are not satisfied merely by the API, Dockerfile or this benchmark. Dataset rights are not established for commercial redistribution.
