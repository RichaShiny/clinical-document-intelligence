# Operations and deployment boundaries

## Supported local workflow

Run preparation, training, encoder download and the biomedical experiment from the README before starting the API. `runtime/artifacts/manifest.json` records the chosen model. The selected entity extractor is recorded separately from the sentence classifier. Legacy bundles default to CRF; promoted biomedical token bundles also require their registered delta checkpoint. The service verifies its trusted artifact hashes and, when needed, encoder hashes before loading. Missing/corrupt artifacts yield readiness 503. Never deserialize arbitrary user-supplied joblib/pickle models. Matching hashes are not an authenticity signature.

`ADE_API_TOKEN` is required at startup. Source text is processed in memory; access logs and operational counters do not include request bodies. Reviewer rationales are explicitly persisted and may themselves be sensitive, so protect the review store. The service does not implement deletion/retention automation. `ADE_REVIEW_DB` optionally relocates the SQLite file.

The default server binds to loopback in the README. There is no public hosting or third-party inference service. Model initialization has a cold-start cost; health readiness is separate from process liveness. Inference handles at most 20,000 characters and 200 heuristic sentences. An ASGI middleware bounds both fixed and chunked request bodies to 128 KiB. CPU inference uses four PyTorch threads, one worker in the example, and sequential batches of up to 24 biomedical sentences.

## Container workflow

The Dockerfile runs as non-root and installs tested direct versions, using CPU PyTorch wheels. Build the image after training if you want the measured report included. Raw data and weights are excluded from the build context.

```bash
docker build -t clinical-ade:local .
docker run --rm -p 127.0.0.1:8000:8000 --env ADE_API_TOKEN \
  --env ADE_REVIEW_DB=/app/reviews/reviews.sqlite \
  --mount type=bind,src="$PWD/runtime/artifacts",dst=/app/runtime/artifacts,readonly \
  --mount type=bind,src="$PWD/runtime/biomedbert",dst=/app/runtime/biomedbert,readonly \
  --mount type=bind,src="$PWD/runtime/test_predictions.json",dst=/app/runtime/test_predictions.json,readonly \
  --mount type=volume,src=ade-reviews,dst=/app/reviews \
  clinical-ade:local
```

The Docker daemon was unavailable during local development, so no successful local image build/run is claimed. CI contains an image-build check; inspect its actual result on the pull request.

## Monitoring and changes

`/metrics` exposes request/error counts and cumulative request seconds, authenticated with the same token. It is basic process telemetry, not a full drift-monitoring platform. Dataset shift, score-distribution drift, new drug terminology, calibration decay and reviewer disagreement require a future monitoring program with outcome labels and agreed thresholds.

To update a model, train into a separate directory, validate manifests and held-out results, run the API integration tests, then restart against the reviewed artifact bundle. Preserve the previous bundle to roll back. Do not swap files under a running service. No continuous learning occurs from reviewer decisions.

Before shared/Internet deployment add managed identity and per-user authorization, TLS termination, rate/concurrency controls, audit ownership, encrypted storage, retention policy and organizational security review. SQLite and a single local bearer token are a local research setup, not multi-tenant production infrastructure. Production-oriented here refers to the implemented packaging/serving discipline, not clinical or operational certification.

### Partial sentence coverage

Biomedical sentence classification and token extraction each use a 128-token
input limit, including special tokens. `/v1/analyze` now returns per-sentence
`coverage.classification` and `coverage.entity_extraction` objects with
`input_tokens`, `max_tokens`, and `truncated`. Counts use the corresponding
model's tokenizer before truncation. A null stage means this bounded biomedical
stage did not run (another backend, or extraction skipped by sentence gating).
It does not certify full coverage for other backends.

Per-sentence `warnings` lists `classification_truncated` and/or
`entity_extraction_truncated` when applicable. The workbench displays these
warnings beside the full source sentence. The returned score and spans still
come from the original 128-token inference; warnings neither recalibrate the
score nor recover omitted evidence. A lower-priority sentence can therefore
carry a classification warning. Review the complete source manually when
coverage is partial. Sentence and span offsets remain in the original input.

Full token counting adds tokenization work but no additional encoder inference.
This change does not implement sliding windows or establish that truncation
caused the sentence-length differences in the error analysis.
