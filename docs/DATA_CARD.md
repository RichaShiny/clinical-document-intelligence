# Data card

## Provenance and scope

All committed note text and gold labels are authored in `clinical_intelligence/dataset.py`. No EHR export, hospital record, scraped patient content, or external dataset is included. Gold concepts and assertions are specified alongside each sentence **before** extraction; they are not the baseline's outputs relabeled as truth. This independence prevents circular labeling but does not remove author bias.

There are 75 development notes, 75 template-test notes, and 16 hand-authored challenge notes. Each template split crosses five conditions, five assertion contexts, and three synthetic age groups. Template notes contain four labeled entities. Challenge notes vary in length and coverage. Clinical vocabulary is deliberately limited to five conditions, four medications, two procedures, and follow-up instructions. Abbreviations map to the same concepts.

Synthetic age groups are metadata, not demographics drawn from a population. Every note has `synthetic: true`. IDs encode the split. Notes are in English and represent a narrow outpatient-style demonstration. There are no real dates, names, addresses, patient IDs, or outcome labels. This is not a statistically representative healthcare dataset.

## Files and checks

`data/synthetic_notes.jsonl` contains text and gold entity spans. Character offsets are zero-based, end-exclusive, Python Unicode offsets. `data/manifest.json` records count, generator version and SHA-256. CI checks deterministic regeneration. Development and test use different sentence templates but share vocabulary and much of the document structure; they are not independent clinical cohorts. Each template appears across age groups, introducing correlation. Bootstrap intervals cannot resolve that limitation.

## External data options

Hugging Face hosts datasets with different origins and access conditions. Hosting and a license badge alone do not establish rights to redistribute patient text.

| Resource | Nature | Suitable extension |
|---|---|---|
| [BC5CDR on Hugging Face](https://huggingface.co/datasets/tner/bc5cdr) | Annotated biomedical literature; chemicals and diseases | A separate NER benchmark; not clinical-note assertion evaluation |
| [Asclepius](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) | Synthetic clinical notes | More varied synthetic development examples; not real patient validation |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | Real deidentified clinical notes, credentialed access | Local clinical-note evaluation after training, credentialing and DUA |

Before adding a corpus: verify original source and terms, preserve the official split, define task and label mappings, check duplicates at document/patient level, and document whether entities, assertions or summaries have independently verified labels. MIMIC notes and patient-level derived outputs must not be added to this public repository. Public availability is not blanket authorization for redistribution or remote-model processing. No external dataset integration or validation is claimed in this version.

Source pages checked October 16, 2026.
