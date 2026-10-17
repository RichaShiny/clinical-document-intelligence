# Local LLM extraction: span alignment experiment

## Setup

The measured run used the installed `llama3.1:latest` model (8B, Q4_K_M) through local Ollama. Exact model digest, runtime version, architecture, prompt/schema hash and capture time are in `reports/llm_environment.json`. The script selected 15 test notes with seed 42 and all 16 challenge notes. No gold annotations were sent to the model. Prompt temperature was zero, with seed 42.

## Observation

The LLM frequently emitted full sentences as evidence despite a prompt requesting mention-level quotes. The initial validator required the quote itself to exactly match a known alias. It rejected 78 model entities. This drove strict-v1 recall down: test F1 0.0328 and challenge F1 0.1667. Rejected quotes that are present in the source are not evidence of a medical hallucination.

## Improvement

Normalized-v2 still requires an exact, unique source quote. It finds exactly one bounded alias for the model's declared canonical concept inside that quote and derives offsets from the source. If no alias or multiple matching aliases occur, the entity is rejected. For follow-up instructions it derives canonical text from a quoted follow-up sentence, removing only a final period. It preserves the model's assertion label; it does not use gold labels or the rule-based assertion system to correct that label.

Raw model field values are preserved in `reports/llm_raw.json`; accepted entities were reconstructed from their original four fields and combined with rejected entities. Entity order may differ but the metric is order-independent. A replay requires no inference and is deterministic:

```bash
python -m clinical_intelligence.replay
```

## Results and interpretation

| Validation policy | Test n=15 | Challenge n=16 |
|---|---:|---:|
| Strict-v1 | 0.0328 F1 | 0.1667 F1 |
| Normalized-v2, same model responses | 0.9244 F1 | 0.8095 F1 |
| Contextual rules, same notes | 1.0000 F1 | 0.7619 F1 |

The replay demonstrates a large integration effect. It does not show that the model learned or that prompting improved. V2 was developed after inspecting evaluated outputs, so these are exploratory results. The small synthetic challenge-set difference is not sufficient evidence to prefer an LLM in a clinical product. The next evaluation must freeze the updated adapter before collecting a new, independently labeled holdout.

Request latency in replayed metrics is the original inference-plus-strict-validation measurement; replay processing time is excluded and labeled. A fresh pipeline run measures the updated validation path. Tokens and zero local API fees refer to the original 31 requests; hardware and electricity costs remain unknown.
