# Research basis and use case

## Problem

Pharmacovigilance literature review requires finding published sentences that may describe medication-related harm, retaining their context, and routing them to a qualified reviewer. This project addresses sentence prioritization and evidence extraction. It does not automatically file regulatory safety reports, adjudicate causation, or recommend treatment.

## Relevant publications

1. **Gurulingappa et al. (2012), Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports.** [Publisher abstract](https://pubmed.ncbi.nlm.nih.gov/22554702/). Establishes the annotated ADE corpus and sentence classification task. Our dataset provenance follows this corpus; our grouped splitting differs from the original cross-validation benchmark, so its reported scores are not directly comparable.
2. **Gurulingappa et al. (2012), Extraction of potential adverse drug events from medical case reports.** [Open paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3599676/). Motivates extracting potential drug/event evidence from literature and explains the corpus's sparse relation annotations. This project extracts associated spans, not a validated causal relation classifier.
3. **Drug Adverse Event Detection Using Text-Based Convolutional Neural Networks (2022).** [Paper](https://www.mdpi.com/2079-9292/11/20/3336). Provides a precedent for neural classification on the ADE corpus and cites the source distribution used for PMID recovery. We implement a different neural architecture (Deep Averaging Network) and document the actual comparison rather than claiming a replication.
4. **Gu et al., Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing.** [Paper](https://arxiv.org/abs/2007.15779), [official BiomedBERT model card](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext). Motivates biomedical-specific representations. Our encoder is frozen; a classifier is trained on ADE labels. This is feature-based transfer learning, not biomedical encoder fine-tuning.

## Design choices

We compare lexical features, a neural model trained from scratch, and domain-specific pretrained representations. Supervised learning is a better fit than unvalidated generation for this binary, labeled task. A CRF adds structured spans with explicit offsets; its weaknesses remain separately visible. Every review candidate preserves the original sentence instead of generating a new medical claim.

More complex models do not automatically win: on the measured tuning split the scratch PyTorch model trails the lexical baseline. BiomedBERT performs best among these candidates. No state-of-the-art or published-paper superiority claim is made. Pretraining may have included source papers, so document-separated supervised evaluation does not establish absence of pretraining overlap.
