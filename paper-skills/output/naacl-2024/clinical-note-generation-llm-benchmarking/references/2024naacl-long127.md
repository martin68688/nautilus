---
title: "Principles from Clinical Research for NLP Model Generalization"
source: "https://aclanthology.org/2024.naacl-long.127/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'clinical-note-generation-llm-benchmarking']
tags: ['generalization', 'clinical-research', 'robustness', 'evaluation']
venue: "NAACL 2024"
tldr: "Applying principles from clinical research to improve and assess the generalization of NLP models beyond standard test sets."
---

# Principles from Clinical Research for NLP Model Generalization

**Source**: [https://aclanthology.org/2024.naacl-long.127/](https://aclanthology.org/2024.naacl-long.127/)

**TLDR**: Applying principles from clinical research to improve and assess the generalization of NLP models beyond standard test sets.

## Abstract

AbstractThe NLP community typically relies on performance of a model on a held-out test set to assess generalization. Performance drops observed in datasets outside of official test sets are generally attributed to “out-of-distribution” effects. Here, we explore the foundations of generalizability and study the factors that affect it, articulating lessons from clinical studies. In clinical research, generalizability is an act of reasoning that depends on (a) *internal validity* of experiments to ensure controlled measurement of cause and effect, and (b) *external validity* or transportability of the results to the wider population. We demonstrate how learning spurious correlations, such as the distance between entities in relation extraction tasks, can affect a model’s internal validity and in turn adversely impact generalization. We, therefore, present the need to ensure internal validity when building machine learning models in NLP. Our recommendations also apply to generative large language models, as they are known to be sensitive to even minor semantic preserving alterations. We also propose adapting the idea of *matching* in randomized controlled trials and observational studies to NLP evaluation to measure causation.