---
title: "Fact Checking Beyond Training Set"
source: "https://aclanthology.org/2024.naacl-long.124/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-attribution-verification']
tags: ['fact-checking', 'generalization']
venue: "NAACL 2024"
tldr: "Fact-checking models suffer performance drops on claims outside their training distribution."
---

# Fact Checking Beyond Training Set

**Source**: [https://aclanthology.org/2024.naacl-long.124/](https://aclanthology.org/2024.naacl-long.124/)

**TLDR**: Fact-checking models suffer performance drops on claims outside their training distribution.

## Abstract

AbstractEvaluating the veracity of everyday claims is time consuming and in some cases requires domain expertise. We empirically demonstrate that the commonly used fact checking pipeline, known as the retriever-reader, suffers from performance deterioration when it is trained on the labeled data from one domain and used in another domain. Afterwards, we delve into each component of the pipeline and propose novel algorithms to address this problem. We propose an adversarial algorithm to make the retriever component robust against distribution shift. Our core idea is to initially train a bi-encoder on the labeled source data, and then, to adversarially train two separate document and claim encoders using unlabeled target data. We then focus on the reader component and propose to train it such that it is insensitive towards the order of claims and evidence documents. Our empirical evaluations support the hypothesis that such a reader shows a higher robustness against distribution shift. To our knowledge, there is no publicly available multi-topic fact checking dataset. Thus, we propose a simple automatic method to re-purpose two well-known fact checking datasets. We then construct eight fact checking scenarios from these datasets, and compare our model to a set of strong baseline models, including recent domain adaptation models that use GPT4 for generating synthetic data.