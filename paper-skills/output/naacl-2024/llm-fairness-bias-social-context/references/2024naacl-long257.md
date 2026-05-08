---
title: "Mitigating Bias for Question Answering Models by Tracking Bias Influence"
source: "https://aclanthology.org/2024.naacl-long.257/"
categories: ['social-bias-mitigation-in-language-models', 'llm-fairness-bias-social-context']
tags: ['bias-mitigation', 'question-answering', 'debiasing']
venue: "NAACL 2024"
tldr: "Proposes a method to mitigate bias in QA models by tracking and reducing the influence of biased training instances."
---

# Mitigating Bias for Question Answering Models by Tracking Bias Influence

**Source**: [https://aclanthology.org/2024.naacl-long.257/](https://aclanthology.org/2024.naacl-long.257/)

**TLDR**: Proposes a method to mitigate bias in QA models by tracking and reducing the influence of biased training instances.

## Abstract

AbstractModels of various NLP tasks have been shown to exhibit stereotypes, and the bias in the question answering (QA) models is especially harmful as the output answers might be directly consumed by the end users. There have been datasets to evaluate bias in QA models, while bias mitigation technique for the QA models is still under-explored. In this work, we propose BMBI, an approach to mitigate the bias of multiple-choice QA models. Based on the intuition that a model would lean to be more biased if it learns from a biased example, we measure the bias level of a query instance by observing its influence on another instance. If the influenced instance is more biased, we derive that the query instance is biased. We then use the bias level detected as an optimization objective to form a multi-task learning setting in addition to the original QA task. We further introduce a new bias evaluation metric to quantify bias in a comprehensive and sensitive way. We show that our method could be applied to multiple QA formulations across multiple bias categories. It can significantly reduce the bias level in all 9 bias categories in the BBQ dataset while maintaining comparable QA accuracy.