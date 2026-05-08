---
title: "Metaphor Detection with Context Enhancement and Curriculum Learning"
source: "https://aclanthology.org/2024.naacl-long.149/"
categories: ['metaphor-analysis-political-framing', 'contrastive-and-generative-representation-learning']
tags: ['metaphor-detection', 'curriculum-learning', 'context-enhancement']
venue: "NAACL 2024"
tldr: "Improves metaphor detection by enhancing context semantics and using curriculum learning to address data imbalance."
---

# Metaphor Detection with Context Enhancement and Curriculum Learning

**Source**: [https://aclanthology.org/2024.naacl-long.149/](https://aclanthology.org/2024.naacl-long.149/)

**TLDR**: Improves metaphor detection by enhancing context semantics and using curriculum learning to address data imbalance.

## Abstract

AbstractMetaphor detection is a challenging task for natural language processing (NLP) systems. Previous works failed to sufficiently utilize the internal and external semantic relationships between target words and their context. Furthermore, they have faced challenges in tackling the problem of data sparseness due to the very limited available training data. To address these two challenges, we propose a novel model called MiceCL. By leveraging the difference between the literal meaning of the target word and the meaning of the sentence as the sentence external difference, MiceCL can better handle the semantic relationships. Additionally, we propose a curriculum learning framework for automatically assessing difficulty of the sentence with a pre-trained model. By starting from easy examples and gradually progressing to more difficult ones, we can ensure that the model will not deal with complex data when its ability is weak so that to avoid wasting limited data. Experimental results demonstrate that MiceCL achieves competitive performance across multiple datasets, with a significantly improved convergence speed compared to other models.