---
title: "Ethos: Rectifying Language Models in Orthogonal Parameter Space"
source: "https://aclanthology.org/2024.findings-naacl.132/"
categories: ['llm-backdoor-attacks-defense', 'llm-alignment-safety-detoxification']
tags: ['alignment', 'detoxification', 'parameter-efficient']
venue: "NAACL 2024"
tldr: "Presents an efficient method to rectify language models by editing weights in an orthogonal parameter space to reduce bias/toxicity and protect privacy."
---

# Ethos: Rectifying Language Models in Orthogonal Parameter Space

**Source**: [https://aclanthology.org/2024.findings-naacl.132/](https://aclanthology.org/2024.findings-naacl.132/)

**TLDR**: Presents an efficient method to rectify language models by editing weights in an orthogonal parameter space to reduce bias/toxicity and protect privacy.

## Abstract

AbstractLanguage models (LMs) have greatly propelled the research on natural language processing. However, LMs also raise concerns regarding the generation of biased or toxic content and the potential disclosure of private information from the training dataset. In this work, we present a new efficient approach, Ethos, that rectifies LMs to mitigate toxicity and bias in outputs and avoid privacy leakage. Ethos is built on task arithmetic. However, unlike current task arithmetic algorithms, Ethos distinguishes general beneficial and undesired knowledge when reconstructing task vectors. Specifically, Ethos first obtains a set of principal components from the pre-trained models using singular value decomposition. Then, by projecting the task vector onto principal components, Ethos separates the principal components that encode general from those associated with undesired knowledge. Ethos performs forgetting or unlearning by only negating the task vector with undesired knowledge, thereby minimizing collateral damage on general model utility. We demonstrate the efficacy of our approach on three different tasks: bias, toxicity, and memorization unlearning. Evaluations show Ethos is more effective in removing undesired knowledge while maintaining the overall model performance compared to current task arithmetic methods.