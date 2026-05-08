---
title: "ExpertQA: Expert-Curated Questions and Attributed Answers"
source: "https://aclanthology.org/2024.naacl-long.167/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['expert-qa', 'attribution', 'verification']
venue: "NAACL 2024"
tldr: "ExpertQA is a dataset of expert-curated questions with attributed answers to improve factual correctness and source verification in language models."
---

# ExpertQA: Expert-Curated Questions and Attributed Answers

**Source**: [https://aclanthology.org/2024.naacl-long.167/](https://aclanthology.org/2024.naacl-long.167/)

**TLDR**: ExpertQA is a dataset of expert-curated questions with attributed answers to improve factual correctness and source verification in language models.

## Abstract

AbstractAs language models are adopted by a more sophisticated and diverse set of users, the importance of guaranteeing that they provide factually correct information supported by verifiable sources is critical across fields of study. This is especially the case for high-stakes fields, such as medicine and law, where the risk of propagating false information is high and can lead to undesirable societal consequences. Previous work studying attribution and factuality has not focused on analyzing these characteristics of language model outputs in domain-specific scenarios. In this work, we conduct human evaluation of responses from a few representative systems along various axes of attribution and factuality, by bringing domain experts in the loop. Specifically, we collect expert-curated questions from 484 participants across 32 fields of study, and then ask the same experts to evaluate generated responses to their own questions. In addition, we ask experts to improve upon responses from language models. The output of our analysis is ExpertQA, a high-quality long-form QA dataset with 2177 questions spanning 32 fields, along with verified answers and attributions for claims in the answers.