---
title: "Asking More Informative Questions for Grounded Retrieval"
source: "https://aclanthology.org/2024.findings-naacl.276/"
categories: ['language-grounded-embodied-navigation', 'spatiotemporal-language-representation-grounding']
tags: ['interactive-qa', 'grounded-retrieval', 'informative-questions']
venue: "NAACL 2024"
tldr: "Enables models in a grounded image identification task to ask more informative, non-polar questions."
---

# Asking More Informative Questions for Grounded Retrieval

**Source**: [https://aclanthology.org/2024.findings-naacl.276/](https://aclanthology.org/2024.findings-naacl.276/)

**TLDR**: Enables models in a grounded image identification task to ask more informative, non-polar questions.

## Abstract

AbstractWhen a model is trying to gather information in an interactive setting, it benefits from asking informative questions. However, in the case of a grounded multi-turn image identification task, previous studies have been constrained to polar yes/no questions (White et al., 2021), limiting how much information the model can gain in a single turn. We present an approach that formulates more informative, open-ended questions. In doing so, we discover that off-the-shelf visual question answering (VQA) models often make presupposition errors, which standard information gain question selection methods fail to account for. To address this issue, we propose a method that can incorporate presupposition handling into both question selection and belief updates. Specifically, we use a two-stage process, where the model first filters out images which are irrelevant to a given question, then updates its beliefs about which image the user intends. Through self-play and human evaluations, we show that our method is successful in asking informative open-ended questions, increasing accuracy over the past state-of-the-art by 14%, while resulting in 48% more efficient games in human evaluations.