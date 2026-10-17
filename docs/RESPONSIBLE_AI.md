# Responsible AI and healthcare governance

This is a documented engineering risk assessment for an educational demonstration, not a legal determination or compliance certification. Sources were checked October 16, 2026. Requirements depend on intended use, jurisdiction, entity roles and deployment design.

## Data and privacy

Only original synthetic notes are committed. This project does not anonymize real records and contains no automated PHI detector. Replacing names in a real note is not sufficient to claim HIPAA de-identification. HHS describes Safe Harbor and Expert Determination approaches; any future real-data workflow needs an appropriate determination, authorized access and organizational review. See [HHS de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html).

There is no real-data upload in the dashboard. Optional inference targets a fixed loopback Ollama endpoint. Local processing does not by itself make an environment compliant. Real-data use would require access controls, authorized model processing, retention rules, secure logging, auditability, and relevant agreements; those controls are not delivered here. Never put credentials, PHI, source notes, or patient-level generated outputs into Git history or public issue reports.

## Risk register and implementation evidence

| Risk | Current mitigation/evidence | Residual risk and next step |
|---|---|---|
| Unsupported extraction | Exact quote, vocabulary and schema validation; rejection log | Valid quotes can be mislabeled; clinician-reviewed gold labels needed |
| Negation/experiencer errors | Explicit assertions; challenge tests; source highlighting | Prefix rules and LLMs can fail; preserve mandatory review |
| Missing important facts | False-negative list and recall | Limited vocabulary and extractive summaries omit material |
| Prompt injection in notes | Treat note as untrusted data; fixed system instruction; one synthetic challenge | Prompt instruction is not a security boundary; expand adversarial suite |
| Privacy disclosure | Synthetic-only public fixtures; fixed local inference URL | Real data require a separately governed environment |
| Bias and uneven performance | Slice counts and metrics, with limitations visible | Synthetic age groups cannot measure population fairness |
| Overstated performance | Separate template and challenge results, sample sizes and intervals | Independent clinical dataset and expert study still required |
| Drift and irreproducibility | Data hash, fixed sample seeds, CI | Model/hardware changes can change output; track digests and rerun evaluation |
| Automation bias | Research-only notices and inspectable evidence | No clinical deployment or decision policy implemented |

## Lifecycle mapping

The [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) and [Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) are reference frameworks, not certifications. This project's application is:

- **Govern:** document ownership, intended use, review expectations and release limitations in the model card.
- **Map:** define the extraction task, synthetic data scope and plausible failures.
- **Measure:** evaluate exact errors, uncertainty and coverage on documented splits.
- **Manage:** keep human review, reject invalid model output, and make failures visible. Production incident response, access controls and deployment monitoring remain future work.

## Regulatory scope

The [FDA Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software) addresses software functions and non-device CDS criteria. This project makes no device classification or exemption determination. A future product's actual intended use and functionality require qualified assessment; adding a “research only” label does not settle regulatory status. HIPAA, FDA considerations and NIST guidance address different issues and should not be described as a single compliance checklist.
