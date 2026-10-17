# Clinical Document Intelligence — ADE Evidence Triage

**A real-data machine learning system for prioritizing adverse-drug-event evidence in medical case reports.**

Safety reviewers must find reports that describe possible medication-related harm among large volumes of literature. This project trains and compares models to flag ADE-related sentences, extracts drug/effect mentions, preserves source evidence, and exposes a review API and workbench. It uses **real MEDLINE case-report sentences downloaded from Hugging Face**, not generated clinical notes.

**20,895 deduplicated sentences · 2,972 source reports · 4,172 test sentences · document-grouped splits.**

This is production-oriented research software with an authenticated inference service, versioned artifacts, tests, telemetry, and container configuration. It is **not a clinically validated product, an adverse-event reporting system, or proof of drug causation**.

## Measured model comparison

<!-- RESULTS_START -->

| Model | Test AUPRC | Recall | Precision | Sentences flagged |
|---|---:|---:|---:|---:|
| tfidf_logistic | 0.733 | 90.0% | 45.2% | 40.7% |
| pytorch_dan | 0.581 | 89.4% | 33.1% | 55.3% |
| biomedbert_frozen | 0.824 | 87.6% | 62.0% | 28.9% |

Selected by tuning AUPRC: **biomedbert_frozen**. Ungated CRF exact span micro F1: **0.387**; gated end-to-end span micro F1: **0.448**.

<!-- RESULTS_END -->

Model selection uses tuning-set AUPRC. Probability calibration and the review threshold use two additional, separate splits. Test recall is measured, not assumed to meet the policy target. Span extraction is evaluated separately and remains exploratory.

## The actual models and frameworks

| Component | Model | Framework | What is learned |
|---|---|---|---|
| Lexical baseline | Word/character TF-IDF + logistic regression | scikit-learn | Vocabulary, IDF and classifier coefficients |
| Neural baseline | Deep Averaging Network: 96-dim embeddings → 64-unit MLP | PyTorch | Word embeddings and network weights, from scratch |
| Biomedical comparator and serving candidate | BiomedBERT masked-mean embeddings + logistic head | Hugging Face Transformers, PyTorch, scikit-learn | Classifier and scaling; **encoder is frozen, not fine-tuned** |
| Structured evidence | BIO conditional random field with contextual and train-only lexical features | sklearn-crfsuite / CRFsuite | Sequence labeling weights for ADE-associated DRUG/EFFECT spans |
| Serving | Calibrated selected classifier + gated CRF | FastAPI / Pydantic / Uvicorn | No online learning |

No Ollama dependency, no fabricated benchmark scores, no synthetic training dataset. Unit-test fixtures are small artificial inputs used only to verify software behavior.

## Reproduce the full experiment

Python 3.13 is the validated environment. Run these commands from the repository root. Data, model weights, embeddings and reviewer records stay in ignored `runtime/`.

```bash
git clone https://github.com/RichaShiny/clinical-document-intelligence.git
cd clinical-document-intelligence
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

# Real corpus download, provenance recovery, deduplication, grouped splits
python -m clinical_ade.data

# Train lexical + PyTorch models, calibrators, review policies and CRF
python -m clinical_ade.train

# Download a pinned biomedical encoder, then train/evaluate its classifier
python -m clinical_ade.download_model
python -m clinical_ade.biomed

pytest -q
```

The encoder download is about 440 MB. Training and the embedding pass were actually executed locally on CPU; this is not a GPU-training claim. `requirements.txt` records the tested direct package versions; `reports/metrics.json` records the measured environment. Hardware and transitive dependency differences can change runtimes and small numerical results. The embedding cache is keyed by prepared data hash, encoder revision and truncation length.

Rerunning `clinical_ade.train` produces a baseline-only bundle; rerun `clinical_ade.biomed` afterwards to include and potentially select the biomedical model. Test reports are regenerated; preserve the committed report if comparing experiments.

## Run the service and reviewer workbench

Choose your own local secret and export it without committing it:

```bash
export ADE_API_TOKEN='replace-with-your-local-secret'
uvicorn clinical_ade.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000**, enter the token, and connect. The workbench can:

- Analyze literature text using the selected trained model, showing exact sentence/mention offsets.
- Preserve evidence sentences as an extractive summary of review candidates.
- Inspect held-out real sentences with PubMed source links and model errors.
- Compare measured models, review workload and document-cluster confidence intervals.
- Save relevant/not-relevant/uncertain judgments to SQLite, keyed by document hash and model version.

Submitted source documents are not stored by the analysis endpoint. Explicit reviewer decisions and rationales are stored locally. Tokens remain in the current page's input; they are not written to browser storage. No external model API receives the submitted text.

| Endpoint | Purpose |
|---|---|
| `POST /v1/analyze` | Inference on up to 20,000 characters / 200 heuristic sentences |
| `GET /v1/evaluation` | Paginated held-out cases from the local evaluated run |
| `GET /v1/report` | Aggregate model/evaluation report |
| `POST /v1/reviews` | Persist a reviewer annotation |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Loaded model readiness; returns 503 for invalid/missing artifacts |
| `GET /metrics` | Request/error/latency counters without source text |

All `/v1/*` routes and `/metrics` require `Authorization: Bearer <token>`. FastAPI request schemas are available at `/docs`. Artifact hashes are verified before deserializing trusted local bundles; hashes detect corruption and do not make arbitrary pickle files safe.

## Data quality and evaluation design

The Hugging Face ADE corpus has 23,516 rows, including duplicate sentence rows for multiple drug–effect relations. We remove **2,621 duplicate normalized rows**. The Hugging Face classification table omits document IDs, so a pinned source distribution recovers PubMed IDs through exact normalized-text matching; **all retained sentences map to source documents**.

A sentence appearing in multiple documents connects their PubMed IDs into the same split group. Stratified group folds allocate roughly 50% to training, 10% to tuning, 10% to calibration, 10% to review-policy selection, and 20% to testing. Every source document and normalized sentence is confined to one split. Published results with different splits are not directly comparable.

Exact source matching aligns drug/effect spans. Ambiguous repeated mentions and overlapping BIO spans are documented rather than silently treated as reliable labels. [The data card](docs/DATA_CARD.md) explains what the annotations do and do not cover.

## Repository structure

```text
src/clinical_ade/   ingestion, ML training, calibration, CRF, inference and API
reports/           measured aggregate results, dataset and artifact manifests
web/               operational review workbench, backed by the inference API
tests/             unit, API, real-data lineage and trained-bundle integration checks
docs/              research, methodology, model card, operations, governance
runtime/           ignored downloaded data, embeddings, models and review DB
Dockerfile         non-root service image; mounted local artifacts
```

[Research and design](docs/RESEARCH.md) · [Evaluation methodology](docs/METHODOLOGY.md) · [Model card](docs/MODEL_CARD.md) · [Operations](docs/OPERATIONS.md) · [Responsible AI](docs/RESPONSIBLE_AI.md)

## Data/model rights

The ADE Hugging Face card lists the dataset license as **unknown**. This repository distributes code, aggregate measurements and provenance hashes, **not the case-report text or trained ADE artifacts**. Downloaded data retain their source terms. Do not infer commercial data rights from public accessibility. The pretrained BiomedBERT model has its own MIT model license; the repository's [MIT license](LICENSE) applies to project code only.

Dataset: [ADE corpus on Hugging Face](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2). Research: [Gurulingappa et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3599676/). Biomedical encoder: [Microsoft BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext).
