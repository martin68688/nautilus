---
title: "Addressing Both Statistical and Causal Gender Fairness in NLP Models"
source: "https://aclanthology.org/2024.findings-naacl.38/"
categories: ['social-bias-mitigation-in-language-models', 'llm-fairness-bias-social-context']
tags: ['fairness', 'counterfactual-augmentation', 'gender-bias']
venue: "NAACL 2024"
tldr: "A method using counterfactual data augmentation addresses both statistical and causal gender fairness in NLP models."
---

# Addressing Both Statistical and Causal Gender Fairness in NLP Models

**Source**: [https://aclanthology.org/2024.findings-naacl.38/](https://aclanthology.org/2024.findings-naacl.38/)

**TLDR**: A method using counterfactual data augmentation addresses both statistical and causal gender fairness in NLP models.

## Abstract

AbstractStatistical fairness stipulates equivalent outcomes for every protected group, whereas causal fairness prescribes that a model makes the same prediction for an individual regardless of their protected characteristics. Counterfactual data augmentation (CDA) is effective for reducing bias in NLP models, yet models trained with CDA are often evaluated only on metrics that are closely tied to the causal fairness notion; similarly, sampling-based methods designed to promote statistical fairness are rarely evaluated for causal fairness. In this work, we evaluate both statistical and causal debiasing methods for gender bias in NLP models, and find that while such methods are effective at reducing bias as measured by the targeted metric, they do not necessarily improve results on other bias metrics. We demonstrate that combinations of statistical and causal debiasing techniques are able to reduce bias measured through both types of metrics.