---
title: "Corpus Considerations for Annotator Modeling and Scaling"
source: "https://aclanthology.org/2024.naacl-long.59/"
categories: ['llm-backdoor-attacks-defense', 'human-judgment-embedding-adjustment']
tags: ['annotation', 'subjective-tasks', 'perspective-modeling', 'scaling']
venue: "NAACL 2024"
tldr: "Discusses corpus considerations for modeling individual annotator perspectives and scaling annotation in subjective NLP tasks."
---

# Corpus Considerations for Annotator Modeling and Scaling

**Source**: [https://aclanthology.org/2024.naacl-long.59/](https://aclanthology.org/2024.naacl-long.59/)

**TLDR**: Discusses corpus considerations for modeling individual annotator perspectives and scaling annotation in subjective NLP tasks.

## Abstract

AbstractRecent trends in natural language processing research and annotation tasks affirm a paradigm shift from the traditional reliance on a single ground truth to a focus on individual perspectives, particularly in subjective tasks. In scenarios where annotation tasks are meant to encompass diversity, models that solely rely on the majority class labels may inadvertently disregard valuable minority perspectives. This oversight could result in the omission of crucial information and, in a broader context, risk disrupting the balance within larger ecosystems. As the landscape of annotator modeling unfolds with diverse representation techniques, it becomes imperative to investigate their effectiveness with the fine-grained features of the datasets in view. This study systematically explores various annotator modeling techniques and compares their performance across seven corpora. From our findings, we show that the commonly used user token model consistently outperforms more complex models. We introduce a composite embedding approach and show distinct differences in which model performs best as a function of the agreement with a given dataset. Our findings shed light on the relationship between corpus statistics and annotator modeling performance, which informs future work on corpus construction and perspectivist NLP.