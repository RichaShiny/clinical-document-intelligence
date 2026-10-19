# Evaluation protocol

## Splits and model fitting

The seed-42 split uses stratified, connected-document-group folds: folds 0–4 train, 5 tune, 6 calibration, 7 review-policy selection, and 8–9 test. All normalized sentences and all source PMIDs stay in one split. It is not a chronological or prospective split. The feature vocabulary, IDF, neural vocabulary, lexical CRF features, and feature scaler fit only training data.

TF-IDF C values (.25, 1, 4) are chosen by tune AUPRC. The PyTorch model trains for 20 epochs with AdamW, seed 42, batch 128; its checkpoint is chosen by tune AUPRC. BiomedBERT's frozen representation uses attention-masked mean pooling, max 128 subword tokens, and a standardized logistic head (C .01, .1, 1, chosen on tune). The encoder revision is pinned. No sentence-encoder fine-tuning or GPU run is claimed. The [entity extension](ENTITY_EXTRACTION.md) separately fine-tunes the last two encoder blocks and a BIO tagging head on CPU.

The CRF compares all training sentences against a train-only balanced sample. Its sampling policy is selected by tune exact span F1. Training-derived drug/effect token membership is a feature, not a test-derived dictionary. Single BIO decoding cannot represent all nested spans. Relation-level causality is not evaluated.

Each model's probabilities receive logistic sigmoid calibration on the calibration split. The largest threshold achieving at least 90% empirical recall on the separate policy split is selected. The server's default model is the candidate with the highest tuning AUPRC. Test labels do not enter these fitting or selection functions. Aggregate test results were inspected during engineering development, including span-extractor feature development; this is a development benchmark, not a preregistered one-shot test or independent external validation.

## Reported outcomes

Primary ranking measure: average precision (AUPRC implementation in scikit-learn). Secondary measures: AUROC, threshold precision/recall/F1, review fraction, Brier score and 10-bin expected calibration error. The confusion matrix has rows true [negative, positive] and columns predicted [negative, positive]. Review fraction measures sentences flagged in this corpus; it is not a measured time saving or patient benefit.

A no-skill average-precision reference is the positive prevalence. Review threshold targets do not guarantee held-out recall. In the measured biomedical model, test recall falls below the 90% policy target; the report preserves that failure. A clinical release would need its own prespecified acceptance thresholds and independent validation.

We bootstrap connected document groups 300 times with seed 42, retaining all sentences for each sampled group, and report empirical 95% intervals for AUPRC, recall and review fraction. This respects within-paper dependence better than row bootstrap but does not capture population shift or annotation uncertainty. Short/long text slices are operational stress checks, not demographic fairness analysis. Reliable protected-attribute metadata are unavailable.

Entity metrics match exact type/start/end sets. Ungated extractor scores isolate extraction quality. Gated metrics measure the combined selected triage model and extractor; missed triage sentences therefore reduce extraction recall. Citation offsets are mechanically verifiable, but a correct source quote does not prove the medical assertion.

## Reproducibility and limits

Hashes cover raw source files and prepared rows. A pinned encoder revision and input hash identify cached embeddings. Saved raw data and per-sentence predictions remain local. The artifact manifest records model and encoder file digests; model code and reports are committed.

There are no synthetic augmentation records, inferred demographic labels, fabricated outcomes, or LLM-generated gold annotations. Remaining limits include selected case-report bias, approximate sentence segmentation, 128-token encoder truncation, sparse/nested entity annotation, near-duplicate paraphrases, potential pretrained-corpus overlap, and lack of an independent contemporary corpus.
