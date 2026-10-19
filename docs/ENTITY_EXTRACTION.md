# Biomedical token extraction experiment

The CRF baseline misses many ADE-associated drug/effect spans. This experiment compares it with a BiomedBERT token classifier while preserving the existing sentence classifier, data preparation, document groups, tuning split and test split.

## Measured results

| Extractor | Tuning exact span F1 | Test exact span F1 |
|---|---:|---:|
| Existing CRF | 0.401 | 0.387 |
| Partially fine-tuned BiomedBERT | 0.577 | 0.581 |

Selected epoch: **2** of 3. Selected extractor: **biomedbert_token**.
The candidate's gated end-to-end test span F1 is **0.626**. The paired test F1 difference is **+0.194**, with a 95% document-group bootstrap interval **[+0.164, +0.217]** (300 resamples). This interval describes this development corpus, not clinical generalization.

## Model and supervision

The CRF uses a word/regex tokenizer, while this tagger uses biomedical subwords. The comparison therefore changes tokenization and representation as well as the learned model; it does not isolate the effect of fine-tuning alone.

The base is the same pinned Microsoft BiomedBERT checkpoint used for sentence features. A five-label BIO head predicts `O`, `B-DRUG`, `I-DRUG`, `B-EFFECT`, `I-EFFECT`. The final two encoder blocks (indices 10 and 11) and classification head are updated; embeddings and the first ten blocks remain frozen. Unlike sentence classification's frozen encoder, the token extractor is **partially fine-tuned**.

Training uses all positive training sentences plus twice as many negative training sentences, sampled with seed 42. No tuning, calibration, policy, or test sentence enters optimization. AdamW uses 2e-5 for the two encoder blocks and 5e-4 for the head, weight decay .01, batch 16, gradient clipping at 1, and three epochs. Training-only token counts set capped inverse-square-root class weights. Checkpoint selection uses tuning exact-span micro F1.

Fast-tokenizer character offsets align gold spans directly to subword tokens. Every subword of a complete span receives BIO supervision; special/padding tokens use the loss ignore index -100. Partial/truncated spans and spans that cannot fit a single BIO sequence are ignored and counted rather than converted to false negative-token supervision. Decoding returns exact slices of the original input. An orphan `I` tag starts a new entity, matching the CRF decoder convention.

## Evaluation and promotion

The tagger replaces the CRF only when its **tuning exact type/start/end micro F1** exceeds the existing CRF's tuning score. The decision is made before candidate test evaluation. Both models are measured on all 4,172 test sentences, with a paired 300-resample document-group bootstrap for the F1 difference, and gated metrics include missed sentence-triage candidates. Type and boundary errors incur both a false positive and false negative. Results and epoch history are stored in `reports/entity_experiment.json`; no raw case-report text or weights are committed.

The test set was already inspected in earlier engineering development. This is a comparison on a development benchmark, not a fresh external evaluation or evidence of clinical readiness. Sparse ADE annotations, ambiguous repeated mentions, nested spans, pretrained-text overlap and 128-token truncation remain limitations. No causality, temporal-relation or exhaustive drug/symptom extraction claim is made.

## Run and serve

After the full README training workflow:

```bash
python -m clinical_ade.token_entities --epochs 3
python -m clinical_ade.benchmark_entities
pytest -q
```

The command saves only the changed block/head tensors in `runtime/artifacts/entities_biomedbert_delta.pt`, records the selected extractor in the artifact manifest, and refreshes aggregate metrics and local held-out predictions. Inference reconstructs the pinned base model and applies a strictly checked parameter-name/shape set. The base-encoder and delta digests are verified by the serving bundle before loading.

Old bundles without `selected_extractor` keep the CRF path. A bundle selecting the neural extractor must register its checkpoint; a missing or invalid checkpoint fails readiness instead of silently falling back. Restart the service against the reviewed bundle after training. Neural extraction runs only on sentences above the existing triage threshold and batches those candidates.

Model weights and derived case-level outputs remain local under the dataset-rights policy. A fresh run of the original sentence-training workflow selects the CRF by default; run token-extractor training last to reproduce this extension.

## Latency protocol

`reports/entity_inference_benchmark.json` compares CRF and neural extraction through the same in-process ASGI API protocol on 20 real held-out sentences, after three warm-ups. Isolated temporary manifests select each backend without modifying the serving bundle. Startup and network transport are excluded; the sample is not a concurrency/load test. Both paths use the unchanged triage threshold, so low-priority sentences bypass extraction. This report is not directly comparable to the earlier socket-HTTP benchmark.

The neural extractor holds a separate encoder from sentence classification, increasing serving memory and adding a forward pass for review candidates. The saved delta avoids redistributing a second full base checkpoint; it does not remove that inference cost.
