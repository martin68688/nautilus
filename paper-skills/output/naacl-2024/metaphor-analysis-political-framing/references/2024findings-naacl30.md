---
title: "A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection"
source: "https://aclanthology.org/2024.findings-naacl.30/"
categories: ['metaphor-analysis-political-framing', 'social-bias-mitigation-in-language-models']
tags: ['malicious-content', 'detection', 'generalization', 'social-graph']
venue: "NAACL 2024"
tldr: "Proposes a realistic evaluation setup for generalizing community models on malicious content detection."
---

# A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection

**Source**: [https://aclanthology.org/2024.findings-naacl.30/](https://aclanthology.org/2024.findings-naacl.30/)

**TLDR**: Proposes a realistic evaluation setup for generalizing community models on malicious content detection.

## Abstract

AbstractCommunity models for malicious content detection, which take into account the context from a social graph alongside the content itself, have shown remarkable performance on benchmark datasets. Yet, misinformation and hate speech continue to propagate on social media networks. This mismatch can be partially attributed to the limitations of current evaluation setups that neglect the rapid evolution of online content and the underlying social graph. In this paper, we propose a novel evaluation setup for model generalisation based on our few-shot subgraph sampling approach. This setup tests for generalisation through few labelled examples in local explorations of a larger graph, emulating more realistic application settings. We show this to be a challenging inductive setup, wherein strong performance on the training graph is not indicative of performance on unseen tasks, domains, or graph structures. Lastly, we show that graph meta-learners trained with our proposed few-shot subgraph sampling outperform standard community models in the inductive setup.