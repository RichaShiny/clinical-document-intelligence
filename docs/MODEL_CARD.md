# Model and system card

## Intended use

Educational healthcare data science portfolio and local experimentation on synthetic text. Outputs are review artifacts for a human analyst. Not designed for diagnosis, prescribing, patient communication, triage, insurance decisions, autonomous chart updates, or clinical use.

## Components

- Dictionary NLP v1: hand-authored alias vocabulary, no learned weights.
- Contextual NLP v1: prefix-based assertion heuristics, no learned weights.
- Optional pretrained Llama 3.1 8B via local Ollama: structured extraction, no fine-tuning. The model remains subject to its own license. A pinned digest and environment details for the measured run are recorded separately when available.
- Sentence selection: deterministic extractive summaries, not an abstractive generation model.

The code uses Python's standard library; evaluation does not depend on a training framework. The dashboard is plain HTML/CSS/JavaScript. No PyTorch training, scikit-learn classifier, spaCy clinical model, BioBERT, or ClinicalBERT is claimed.

## Evidence and known weaknesses

Full baseline results and optional measured LLM results are in `reports/metrics.json`. All baseline notes and evaluated LLM notes have inspectable prediction/error details in the dashboard exports. Baseline contextual F1 is 1.000 on 75 templated test notes and 0.762 on 16 challenge notes. This gap illustrates narrow test coverage. The synthetic dataset cannot establish safety, performance on hospital documents, calibration, fairness or clinical benefit.

Rules miss post-mention cues (“Pneumonia was ruled out”), misread “No change in asthma treatment,” and lack complex experiencer/temporal reasoning. LLM quotes can be genuine while assertion labels are wrong. Accepted LLM entities are not independently verified clinical facts. Missing or rejected mentions can remove important material from summary cards.

The local LLM is nondeterministic across environments despite a zero temperature and fixed seed. The generator's data and task labels are limited, and pretrained model training overlap is unknown. There is no calibrated confidence score or automated clinical acceptance threshold.

## Release gate for any future clinical study

A designated clinical owner must establish the intended use, select an authorized dataset, independently adjudicate labels, specify risk-based thresholds before evaluation, review subgroup coverage and harmful errors, and establish monitoring and rollback procedures. This repository does not implement or satisfy that gate.
