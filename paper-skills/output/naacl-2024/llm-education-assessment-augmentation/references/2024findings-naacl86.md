---
title: "An Effective Automated Speaking Assessment Approach to Mitigating Data Scarcity and Imbalanced Distribution"
source: "https://aclanthology.org/2024.findings-naacl.86/"
categories: ['speech-language-processing-multilingual', 'llm-education-assessment-augmentation']
tags: ['speech-assessment', 'low-resource', 'imbalanced-data']
venue: "NAACL 2024"
tldr: "Proposes an automated speaking assessment method using SSL and data augmentation to handle scarcity and imbalance."
---

# An Effective Automated Speaking Assessment Approach to Mitigating Data Scarcity and Imbalanced Distribution

**Source**: [https://aclanthology.org/2024.findings-naacl.86/](https://aclanthology.org/2024.findings-naacl.86/)

**TLDR**: Proposes an automated speaking assessment method using SSL and data augmentation to handle scarcity and imbalance.

## Abstract

AbstractAutomated speaking assessment (ASA) typically involves automatic speech recognition (ASR) and hand-crafted feature extraction from the ASR transcript of a learner’s speech. Recently, self-supervised learning (SSL) has shown stellar performance compared to traditional methods. However, SSL-based ASA systems are faced with at least three data-related challenges: limited annotated data, uneven distribution of learner proficiency levels and non-uniform score intervals between different CEFR proficiency levels. To address these challenges, we explore the use of two novel modeling strategies: metric-based classification and loss re-weighting, leveraging distinct SSL-based embedding features. Extensive experimental results on the ICNALE benchmark dataset suggest that our approach can outperform existing strong baselines by a sizable margin, achieving a significant improvement of more than 10% in CEFR prediction accuracy.