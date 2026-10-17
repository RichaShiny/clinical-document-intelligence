# Portfolio walkthrough

## A 90-second explanation

“I built an evaluation lab for turning clinical text into structured, evidence-linked information. I implemented two transparent NLP baselines and integrated a pretrained local LLM through a constrained extraction schema. The project measures exact entity and assertion errors, uses paired bootstrap comparisons, and exposes individual mistakes in a dashboard. I kept the public fixtures synthetic and documented what would change for governed patient-data research.”

Use the measured LLM results only if `reports/metrics.json` records actual LLM execution; name its evaluated sample sizes. Do not claim clinical validation, a trained/fine-tuned model, or experience deploying into a healthcare organization from this demo.

## Demonstration sequence

1. Open the dashboard on contextual/template test. Explain the exact entity definition and why the simple benchmark scores perfectly.
2. Switch to challenge cases and filter to errors. Show “Pneumonia was ruled out” and discuss post-mention assertion scope.
3. Compare dictionary vs contextual behavior on the same note.
4. Select the LLM, if measured, and explain JSON validation, quote provenance, residual semantic errors and the matched-subset baseline.
5. Show the data manifest, automated tests, and responsible AI risk register.

## Questions to prepare for

**Why not just use an LLM?** A transparent baseline establishes cost and error tradeoffs. Compare on identical notes before attributing a gain to the model. Schema adherence cannot prove clinical correctness.

**Why synthetic data?** It makes the public demo immediately reproducible and avoids distributing real patient notes. It is an engineering fixture, not external validation. A next study should use an authorized corpus with independent annotations.

**What would you train?** With sufficient labeled data, an assertion classifier or clinical token-classification model; freeze patient-level splits, compare against these baselines, and perform label-quality and error analysis. That training is not part of this release.

**What does the confidence interval mean?** It reflects resampling within this corpus, not certainty about hospitals or patient outcomes. Repeated templates also violate strong independence assumptions.

**What would block release?** Unresolved harmful errors, inadequate clinical review, unauthorized data processing, unsupported intended-use claims, or missing operational monitoring and incident processes.

## Truthful resume phrasing

“Built a reproducible Python clinical NLP evaluation pipeline with local LLM structured extraction, evidence-linked summaries, bootstrap analysis, an interactive error dashboard, and documented data/AI governance.”

If adding numbers, give dataset sizes and explain that the notes are synthetic. Do not present template-test F1 as real-world accuracy.
