# Dataset card and annotation audit

## Origin and task

The data come from `ade-benchmark-corpus/ade_corpus_v2` on Hugging Face, revision `4ba01c71687dd7c996597042449448ea312126cf`. The classification and drug/ADE relation Parquet files are downloaded directly from that revision. Their original text is from MEDLINE medical case reports, not synthetic EHR notes. The research corpus is described by [Gurulingappa et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3599676/).

The classification target is whether a sentence is ADE-related according to corpus annotations. It is not a causal probability, event severity, patient risk estimate, or a calibrated probability that a medication is unsafe. The corpus is a selected, historical English case-report sample; it is not representative of all literature or healthcare settings.

## Document provenance

The HF classification table has text and labels but no PMID. A pinned [source distribution](https://github.com/trunghlt/AdverseDrugReaction/tree/2851af96e26050cf551a2e1f716ff861f7b1059b/ADE-Corpus-V2) supplies source PubMed IDs. This distribution is also cited by the published TextCNN work linked in the research note. All retained classification sentences match after whitespace normalization and case folding. The pipeline records the unmatched count and excludes unmatched records rather than inventing document IDs.

Exact/normalized duplicate sentences are collapsed; conflicting labels would be removed. Multi-document duplicate sentences connect their source PMIDs into one group using union-find. This stops duplicates linking papers across train/test. Similar paraphrases are not automatically detected; this remains a residual leakage risk. Grouping at case-report PMID does not guarantee unique patients across different publications.

## Actual audit

The pipeline retained 20,895 normalized sentences from 2,972 source documents, forming 2,911 connected document groups. It removed 2,621 repeated rows, found zero label conflicts and zero missing PMID mappings. All file hashes and split counts are in `reports/data_manifest.json`. The cleaned class prevalence is lower than the raw row prevalence because positive sentences often occur once per annotated relation in the raw data.

For entity training, literal drug/effect surfaces are aligned to the sentence when each occurs uniquely. Ambiguous repeated mentions are excluded (725 annotation instances, including repeated relation rows); original HF offsets disagree with unique literal alignment or are missing in 37 aligned instances. Raw source-distribution offsets are document-level and must not be naively interpreted as sentence offsets. Overlapping spans cannot all be represented by a single BIO sequence; the CRF baseline report records excluded training overlaps, and the token-extraction experiment records its own alignment/truncation exclusions.

These exclusions produce a restricted span benchmark. Some excluded or incompletely annotated mentions may be counted as model false positives; this is not an exhaustive drug/symptom NER dataset. Negative sentences do not carry exhaustive drug entity labels. Both entity models target ADE-associated mentions under this corpus's annotation scheme.

## Access, rights and storage

The HF card reports `license: unknown` and has incomplete provenance/licensing fields. Code and aggregate reports are public; raw text, per-sentence predictions, embeddings, trained weights and reviewer annotations are ignored in Git. The pipeline provides reproducible local acquisition without claiming commercial redistribution permission. Any commercial use needs rights review. No private patient-chart access is implied by this public literature corpus.
