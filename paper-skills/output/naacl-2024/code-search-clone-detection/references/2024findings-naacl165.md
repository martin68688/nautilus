---
title: "Best of Both Worlds: A Pliable and Generalizable Neuro-Symbolic Approach for Relation Classification"
source: "https://aclanthology.org/2024.findings-naacl.165/"
categories: ['code-search-clone-detection']
tags: ['neuro-symbolic', 'relation-classification', 'rule-based', 'deep-learning']
venue: "NAACL 2024"
tldr: "This paper introduces a novel neuro-symbolic architecture for relation classification that combines rule-based methods with deep learning."
---

# Best of Both Worlds: A Pliable and Generalizable Neuro-Symbolic Approach for Relation Classification

**Source**: [https://aclanthology.org/2024.findings-naacl.165/](https://aclanthology.org/2024.findings-naacl.165/)

**TLDR**: This paper introduces a novel neuro-symbolic architecture for relation classification that combines rule-based methods with deep learning.

## Abstract

AbstractThis paper introduces a novel neuro-symbolic architecture for relation classification (RC) that combines rule-based methods with contemporary deep learning techniques. This approach capitalizes on the strengths of both paradigms: the adaptability of rule-based systems and the generalization power of neural networks. Our architecture consists of two components: a declarative rule-based model for transparent classification and a neural component to enhance rule generalizability through semantic text matching.Notably, our semantic matcher is trained in an unsupervised domain-agnostic way, solely with synthetic data.Further, these components are loosely coupled, allowing for rule modifications without retraining the semantic matcher.In our evaluation, we focused on two few-shot relation classification datasets: Few-Shot TACRED and a Few-Shot version of NYT29. We show that our proposed method outperforms previous state-of-the-art models in three out of four settings, despite not seeing any human-annotated training data.Further, we show that our approach remains modular and pliable, i.e., the corresponding rules can be locally modified to improve the overall model. Human interventions to the rules for the TACRED relation org:parents boost the performance on that relation by as much as 26% relative improvement, without negatively impacting the other relations, and without retraining the semantic matching component.