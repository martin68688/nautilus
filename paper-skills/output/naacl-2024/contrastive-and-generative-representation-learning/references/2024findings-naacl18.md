---
title: "MiLe Loss: a New Loss for Mitigating the Bias of Learning Difficulties in Generative Language Models"
source: "https://aclanthology.org/2024.findings-naacl.18/"
categories: ['efficient-large-model-training-optimization', 'contrastive-and-generative-representation-learning']
tags: ['loss-function', 'generative-model', 'learning-difficulty']
venue: "NAACL 2024"
tldr: "Proposes a new loss function for generative language models to mitigate bias from varying token-level learning difficulties during pre-training."
---

# MiLe Loss: a New Loss for Mitigating the Bias of Learning Difficulties in Generative Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.18/](https://aclanthology.org/2024.findings-naacl.18/)

**TLDR**: Proposes a new loss function for generative language models to mitigate bias from varying token-level learning difficulties during pre-training.

## Abstract

AbstractGenerative language models are usually pre-trained on large text corpus via predicting the next token (i.e., sub-word/word/phrase) given the previous ones. Recent works have demonstrated the impressive performance of large generative language models on downstream tasks. However, existing generative language models generally neglect an inherent challenge in text corpus during training, i.e., the imbalance between frequent tokens and infrequent ones. It can lead a language model to be dominated by common and easy-to-learn tokens, thereby overlooking the infrequent and difficult-to-learn ones. To alleviate that, we propose a **MiLe Loss** function for **mi**tigating the bias of **le**arning difficulties with tokens. During training, it can dynamically assess the learning difficulty of a to-be-learned token, according to the information entropy of the corresponding predicted probability distribution over the vocabulary. Then it scales the training loss adaptively, trying to lead the model to focus more on the difficult-to-learn tokens. On the Pile dataset, we train generative language models at different scales of 468M, 1.2B, and 6.7B parameters. Experiments reveal that models incorporating the proposed MiLe Loss can gain consistent performance improvement on downstream benchmarks.