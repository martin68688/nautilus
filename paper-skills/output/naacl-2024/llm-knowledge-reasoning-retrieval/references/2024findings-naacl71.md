---
title: "EDEntail: An Entailment-based Few-shot Text Classification with Extensional Definition"
source: "https://aclanthology.org/2024.findings-naacl.71/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization']
tags: ['few-shot', 'entailment', 'text-classification']
venue: "NAACL 2024"
tldr: "Proposes EDEntail, a few-shot text classification method using extensional definitions (example lists) in entailment hypotheses."
---

# EDEntail: An Entailment-based Few-shot Text Classification with Extensional Definition

**Source**: [https://aclanthology.org/2024.findings-naacl.71/](https://aclanthology.org/2024.findings-naacl.71/)

**TLDR**: Proposes EDEntail, a few-shot text classification method using extensional definitions (example lists) in entailment hypotheses.

## Abstract

AbstractFew-shot text classification has seen significant advancements, particularly with entailment-based methods, which typically use either class labels or intensional definitions of class labels in hypotheses for label semantics expression. In this paper, we propose EDEntail, a method that employs extensional definition (EDef) of class labels in hypotheses, aiming to express the semantics of class labels more explicitly. To achieve the above goal, we develop an algorithm to gather and select extensional descriptive words of class labels and then order and format them into a sequence to form hypotheses. Our method has been evaluated and compared with state-of-the-art models on five classification datasets. The results demonstrate that our approach surpasses the supervised-learning methods and prompt-based methods under the few-shot setting, which underlines the potential of using an extensional definition of class labels for entailment-based few-shot text classification. Our code is available at https://github.com/MidiyaZhu/EDEntail.