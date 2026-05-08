---
title: "Source-Free Unsupervised Domain Adaptation for Question Answering via Prompt-Assisted Self-learning"
source: "https://aclanthology.org/2024.findings-naacl.44/"
categories: ['llm-ranking-benchmarking-adaptation']
tags: ['domain-adaptation', 'question-answering', 'prompting', 'self-learning']
venue: "NAACL 2024"
tldr: "Addresses source-free domain adaptation for QA using prompt-assisted self-learning without source data."
---

# Source-Free Unsupervised Domain Adaptation for Question Answering via Prompt-Assisted Self-learning

**Source**: [https://aclanthology.org/2024.findings-naacl.44/](https://aclanthology.org/2024.findings-naacl.44/)

**TLDR**: Addresses source-free domain adaptation for QA using prompt-assisted self-learning without source data.

## Abstract

AbstractThis work addresses source-free domain adaptation (SFDA) for Question Answering (QA), wherein a model trained on a source domain is adapted to unlabeled target domains without additional source data. Existing SFDA methods only focus on the adaptation phase, overlooking the impact of source domain training on model generalizability. In this paper, we argue that source model training itself is also critical for improving the adaptation performance and stability. To this end, we investigate the role of prompt learning as an effective method to internalize domain-agnostic QA knowledge, which can be integrated into source training. After source training, an interactive self-learning strategy is proposed to further fine tune both model and prompt in the model adaptation phase. This leads to the Prompt-Assisted Self-Adaptive Learning (PASAL), an innovative SFDA approach for QA. Empirical evaluation on four benchmark datasets shows that PASAL surpasses existing methods in managing domain gaps and demonstrates greater stability across various target domains, validating the significance of source domain training for effective domain adaptation.