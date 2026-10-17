# Responsible AI and healthcare governance

This is research engineering documentation, not a legal compliance determination. Sources were checked October 16, 2026.

## Scope and accountability

The system prioritizes public medical literature for a human safety reviewer. It makes no patient-care or regulatory decision. Correct extraction does not establish causation, and a probability is not a treatment recommendation. Source evidence and reviewer uncertainty remain visible. Source documents are not persisted by the analysis endpoint; reviewer decisions are stored deliberately and require access/retention controls.

The [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) provides the lifecycle structure: govern intended use and owners; map corpus bias and failure modes; measure ranking, calibration, threshold tradeoffs and extraction; manage review, release gates and rollback. It is guidance, not a certification. This project implements only part of an operational governance program.

## Risk register

| Risk | Implemented control | Remaining gap |
|---|---|---|
| Inflated results from duplicate sentences/documents | Source PMID recovery, deduplication, connected-document grouping and tests | Near-paraphrase and pretrained-corpus overlap remain |
| Missed safety evidence | Policy-target recall, held-out recall and clustered intervals | Test recall can miss target; independent acceptance study required |
| False alarms/reviewer burden | Measured precision and review fraction | No prospective reviewer time/utility study |
| Incorrect drug/effect spans | Separate exact-span evaluation and source offsets | CRF quality is limited; expert verification required |
| False causality | No automatic relation/causality assertion | Users can still overinterpret contextual co-occurrence |
| Unauthorized data use | No raw data/derived case records in Git; pinned download instructions | HF dataset license is unknown; commercial rights not established |
| Data exposure | Bearer authentication, loopback example, bounded requests, no source-body logging | No enterprise identity, encryption-at-rest program or multi-tenant authorization |
| Biased performance | Corpus scope documented; length slices reported | No reliable demographic metadata or fairness assurance |
| Artifact corruption | Hash verification and fail-closed readiness | Digests do not prove origin if attacker changes both data and manifest |
| Model drift | Versioned artifacts and offline reproducible evaluation | No live drift detector, clinical monitoring or incident team |

## Regulatory context

This corpus is published literature, not an authorized private EHR feed. The project does not perform HIPAA de-identification or establish that a future real-patient-data workflow is permissible. [HHS guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) describes HIPAA de-identification approaches; merely replacing identifiers or running a model locally is insufficient to claim compliance.

The [FDA CDS guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software) concerns software functions and intended use. No device/non-device determination, FDA clearance or exemption is claimed here. A future clinical function or regulated reporting workflow needs qualified regulatory assessment. Dataset rights, health privacy obligations, medical device requirements and AI risk guidance are distinct questions.
