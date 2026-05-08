---
title: "Leveraging the Structure of Pre-trained Embeddings to Minimize Annotation Effort"
source: "https://aclanthology.org/2024.naacl-long.387/"
categories: ['contrastive-and-generative-representation-learning', 'multi-label-active-weak-supervision']
tags: ['active_learning', 'representation_learning', 'text_classification']
venue: "NAACL 2024"
tldr: "A method reduces annotation effort for text classification by leveraging the structure of pre-trained LLM embeddings."
---

# Leveraging the Structure of Pre-trained Embeddings to Minimize Annotation Effort

**Source**: [https://aclanthology.org/2024.naacl-long.387/](https://aclanthology.org/2024.naacl-long.387/)

**TLDR**: A method reduces annotation effort for text classification by leveraging the structure of pre-trained LLM embeddings.

## Abstract

AbstractMost current state-of-the-art approaches for text classification are based on fine-tuning the representations computed by large language models (LLMs). This strategy has led to significant improvements in classification performance and contributed to a reduction of the amount of labeled data required for training a model. However, for some challenging classification tasks, providing enough annotations to ensure a reliable classification continues to be the main bottleneck. This is especially true in settings of highly imbalanced class distributions. This paper proposes to tackle this bottleneck by exploiting the structural properties of pre-trained embeddings. We develop a label propagation method that uses pre-trained embeddings to spread information from the labeled samples to nearby samples in the induced space, ensuring the optimal use of annotations. Our approach is simple and relatively low-cost since it only requires computing some distances in the embedded space. We conduct experiments on different text classification datasets showing that the proposed method is efficient and significantly outperforms both self-training and random walk label propagation strategies.