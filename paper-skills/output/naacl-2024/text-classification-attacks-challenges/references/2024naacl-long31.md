---
title: "Query-Efficient Textual Adversarial Example Generation for Black-Box Attacks"
source: "https://aclanthology.org/2024.naacl-long.31/"
categories: ['continual-learning-memory-transfer-llms', 'text-classification-attacks-challenges']
tags: ['adversarial-attacks', 'black-box', 'query-efficiency']
venue: "NAACL 2024"
tldr: "Proposes a query-efficient method for generating textual adversarial examples in black-box attack settings."
---

# Query-Efficient Textual Adversarial Example Generation for Black-Box Attacks

**Source**: [https://aclanthology.org/2024.naacl-long.31/](https://aclanthology.org/2024.naacl-long.31/)

**TLDR**: Proposes a query-efficient method for generating textual adversarial examples in black-box attack settings.

## Abstract

AbstractDeep neural networks for Natural Language Processing (NLP) have been demonstrated to be vulnerable to textual adversarial examples. Existing black-box attacks typically require thousands of queries on the target model, making them expensive in real-world applications. In this paper, we propose a new approach that guides the word substitutions using prior knowledge from the training set to improve the attack efficiency. Specifically, we introduce Adversarial Boosting Preference (ABP), a metric that quantifies the importance of words and guides adversarial word substitutions. We then propose two query-efficient attack strategies based on ABP: query-free attack (ABPfree) and guided search attack (ABPguide). Extensive evaluations for text classification demonstrate that ABPfree generates more natural adversarial examples than existing universal attacks, ABPguide significantly reduces the number of queries by a factor of 10 500 while achieving comparable or even better performance than black-box attack baselines. Furthermore, we introduce the first ensemble attack ABPens in NLP, which gains further performance improvements and achieves better transferability and generalization by the ensemble of the ABP across different models and domains. Code is available at https://github.com/BaiDingHub/ABP.