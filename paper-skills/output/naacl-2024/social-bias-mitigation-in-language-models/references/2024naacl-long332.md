---
title: "Debiasing with Sufficient Projection: A General Theoretical Framework for Vector Representations"
source: "https://aclanthology.org/2024.naacl-long.332/"
categories: ['clinical-nlp-biomedical-applications', 'social-bias-mitigation-in-language-models', 'llm-fairness-bias-social-context']
tags: ['debiasing', 'representation-learning', 'theory']
venue: "NAACL 2024"
tldr: "Proposes a theoretical framework for debiasing vector representations using sufficient projection to remove unwanted social biases."
---

# Debiasing with Sufficient Projection: A General Theoretical Framework for Vector Representations

**Source**: [https://aclanthology.org/2024.naacl-long.332/](https://aclanthology.org/2024.naacl-long.332/)

**TLDR**: Proposes a theoretical framework for debiasing vector representations using sufficient projection to remove unwanted social biases.

## Abstract

AbstractPre-trained vector representations in natural language processing often inadvertently encode undesirable social biases. Identifying and removing unwanted biased information from vector representation is an evolving and significant challenge. Our study uniquely addresses this issue from the perspective of statistical independence, proposing a framework for reducing bias by transforming vector representations to an unbiased subspace using sufficient projection. The key to our framework lies in its generality: it adeptly mitigates bias across both debiasing and fairness tasks, and across various vector representation types, including word embeddings and output representations of transformer models. Importantly, we establish the connection between debiasing and fairness, offering theoretical guarantees and elucidating our algorithm’s efficacy. Through extensive evaluation of intrinsic and extrinsic metrics, our method achieves superior performance in bias reduction while maintaining high task performance, and offers superior computational efficiency.