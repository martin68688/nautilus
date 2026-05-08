---
title: "Capturing Perspectives of Crowdsourced Annotators in Subjective Learning Tasks"
source: "https://aclanthology.org/2024.naacl-long.407/"
categories: ['continual-learning-memory-transfer-llms']
tags: ['annotation', 'subjectivity', 'crowdsourcing']
venue: "NAACL 2024"
tldr: "Proposes a method to capture the diverse perspectives of crowdsourced annotators in subjective classification tasks instead of aggregating via majority vote."
---

# Capturing Perspectives of Crowdsourced Annotators in Subjective Learning Tasks

**Source**: [https://aclanthology.org/2024.naacl-long.407/](https://aclanthology.org/2024.naacl-long.407/)

**TLDR**: Proposes a method to capture the diverse perspectives of crowdsourced annotators in subjective classification tasks instead of aggregating via majority vote.

## Abstract

AbstractSupervised classification heavily depends on datasets annotated by humans. However, in subjective tasks such as toxicity classification, these annotations often exhibit low agreement among raters. Annotations have commonly been aggregated by employing methods like majority voting to determine a single ground truth label. In subjective tasks, aggregating labels will result in biased labeling and, consequently, biased models that can overlook minority opinions. Previous studies have shed light on the pitfalls of label aggregation and have introduced a handful of practical approaches to tackle this issue. Recently proposed multi-annotator models, which predict labels individually per annotator, are vulnerable to under-determination for annotators with few samples. This problem is exacerbated in crowdsourced datasets. In this work, we propose Annotator Aware Representations for Texts (AART) for subjective classification tasks. Our approach involves learning representations of annotators, allowing for exploration of annotation behaviors. We show the improvement of our method on metrics that assess the performance on capturing individual annotators’ perspectives. Additionally, we demonstrate fairness metrics to evaluate our model’s equability of performance for marginalized annotators compared to others.