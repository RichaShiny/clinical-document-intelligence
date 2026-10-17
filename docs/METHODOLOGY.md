# Evaluation methodology

## Systems and hypotheses

The dictionary system finds bounded case-insensitive alias matches and labels them present (follow-up instructions are always planned). The contextual system uses exactly the same extraction vocabulary, then applies prefix rules for family, negation, uncertainty, history and planning. Clause boundaries reset context at semicolons, “but” and “however.” The hypothesis is that assertion handling improves exact entity correctness without changing detection vocabulary.

The optional LLM system uses a JSON schema and a fixed clinical-extraction prompt. It receives no gold labels. Evidence must be an exact unique source quote, its type and status must be valid, and its canonical value must match exactly one bounded supported alias inside that quote. The normalizer aligns the entity span within sentence-level quotes. Follow-up values are derived from a quoted follow-up instruction, with a final period removed. Invalid entities are rejected and retained in per-note audit output. Envelope or transport errors fail the run. Duplicate evidence quotes are rejected rather than ambiguously located. This conservative policy may reduce recall; rejection is not proof the underlying model hallucinated.

## Endpoints

Primary entity correctness is exact `(kind, canonical value, assertion, start, end)`. True positives are set intersections; unmatched predictions and gold entities are false positives and false negatives. A wrong assertion creates one false positive and one false negative. Duplicate identical predictions do not earn additional credit. Micro precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). Empty denominators return zero, explicitly; do not interpret an empty category as perfect performance.

Summary cards reproduce source sentences that contain extracted mentions. Their offsets can be mechanically checked, but clinical completeness, salience and assertion correctness remain unvalidated. There are no ROUGE or hallucination-rate claims for an unimplemented abstractive summarizer. The committed LLM results are a post-hoc replay of saved model responses with normalization v2; original strict-v1 metrics are preserved. Original request latency excludes offline replay time. Quote rejection count is reported separately from semantic error metrics.

## Statistical analysis

We resample notes 500 times with replacement using seed 42, recompute micro F1, and report empirical 2.5%/97.5% bounds. Paired differences resample the same note indices across systems. They describe variation within this constructed corpus. Template reuse and the small challenge set weaken independence and external validity. A collapsed interval at F1=1 reflects a trivial benchmark, not certainty about future clinical performance.

LLM evaluation covers 15 test notes selected by `random.Random(42).sample` and all 16 challenge notes. Matched contextual baseline metrics are included under `llm_comparison`; the main baseline test uses 75 notes. Do not compare these two test populations as if identical. No LLM development scores are fabricated.

Latency is wall-clock sequential time per note, including post-processing and first-call model loading. Median and empirical p95 are measured over each system's evaluated notes. Hardware and cold-start effects matter. Tokens are Ollama-reported counts; API fees are zero for local inference, while compute cost is unknown.

## Product decisions

1. Preserve assertions in structured output: a disease mention is not automatically a current patient diagnosis.
2. Make source evidence inspectable and keep human review in the workflow.
3. Use rules as a transparent baseline and cost reference; model complexity needs demonstrated gains on matched data.
4. Treat challenge failures as development leads, then evaluate changes on a new frozen holdout. Do not tune against this challenge set and reuse its score as an untouched test result.
5. Expand terminology and annotation coverage before deployment claims. Medication dosage, relations, temporal reasoning, labs, coreference and clinical coding are out of scope.

A larger study should use independent annotators with clinician adjudication, patient-level splits, confidence intervals clustered by patient/document, repeated model runs, and expert review of harmful omissions and unsupported assertions.
