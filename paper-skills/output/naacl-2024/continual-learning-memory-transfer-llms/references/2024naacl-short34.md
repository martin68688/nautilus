---
title: "Direct Preference Optimization for Neural Machine Translation with Minimum Bayes Risk Decoding"
source: "https://aclanthology.org/2024.naacl-short.34/"
categories: ['continual-learning-memory-transfer-llms', 'bayes-risk-decoding-and-data-drift']
tags: ['machine-translation', 'preference-optimization', 'minimum-bayes-risk']
venue: "NAACL 2024"
tldr: "Applies Direct Preference Optimization with Minimum Bayes Risk decoding to improve neural machine translation efficiently."
---

# Direct Preference Optimization for Neural Machine Translation with Minimum Bayes Risk Decoding

**Source**: [https://aclanthology.org/2024.naacl-short.34/](https://aclanthology.org/2024.naacl-short.34/)

**TLDR**: Applies Direct Preference Optimization with Minimum Bayes Risk decoding to improve neural machine translation efficiently.

## Abstract

AbstractMinimum Bayes Risk (MBR) decoding can significantly improve translation performance of Multilingual Large Language Models (MLLMs). However, MBR decoding is computationally expensive. We show how the recently developed Reinforcement Learning technique, Direct Preference Optimization (DPO), can fine-tune MLLMs to get the gains of MBR without any additional computation in inference. Our method uses only a small monolingual fine-tuning set and yields significantly improved performance on multiple NMT test sets compared to MLLMs without DPO.