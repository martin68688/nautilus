---
title: "QualEval: Qualitative Evaluation for Model Improvement"
source: "https://aclanthology.org/2024.naacl-long.115/"
categories: ['large-language-model-evaluation-augmentation', 'language-model-evaluation-benchmarks']
tags: ['evaluation', 'qualitative', 'model-improvement']
venue: "NAACL 2024"
tldr: "Advocates for qualitative evaluation to complement quantitative metrics for nuanced model improvement."
---

# QualEval: Qualitative Evaluation for Model Improvement

**Source**: [https://aclanthology.org/2024.naacl-long.115/](https://aclanthology.org/2024.naacl-long.115/)

**TLDR**: Advocates for qualitative evaluation to complement quantitative metrics for nuanced model improvement.

## Abstract

AbstractQuantitative evaluation metrics have been pivotal in gauging the advancements of AI systems like large language models (LLMs).However, due to the intricate nature of real-world tasks, a single scalar to quantify and compare performance trivializes the fine-grained nuances of model behavior. Additionally, metrics do not yield actionable diagnostics for model improvement, thus requiring extensive manual efforts of scientists, involving sifting through vast datasets and attempting hit-or-miss adjustments to training data or setups. In this work, we address the shortcomings of quantitative metrics by proposing QualEval, which uses automated qualitative evaluation as a vehicle for model improvement. QualEval uses a powerful LLM reasoner and our novel flexible linear programming solver to generate human-readable insights that when applied, accelerate model improvement. The insights are supported by a dashboard report with fine-grained visualizations and human-interpretable analyses. We corroborate the faithfulness of QualEval by demonstrating that leveraging its insights, for example, improves the absolute performance of the Llama 2 model by up to 15% points relative on a challenging dialogue task (DialogSum) when compared to baselines. QualEval successfully increases the pace and quality of model development by eliminating the need of arduous manual analysis, thus serving as a data-scientist-in-a-box.