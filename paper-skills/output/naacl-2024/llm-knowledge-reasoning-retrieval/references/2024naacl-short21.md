---
title: "On Retrieval Augmentation and the Limitations of Language Model Training"
source: "https://aclanthology.org/2024.naacl-short.21/"
categories: ['efficient-large-model-training-optimization', 'llm-knowledge-reasoning-retrieval']
tags: ['retrieval-augmentation', 'kNN', 'perplexity']
venue: "NAACL 2024"
tldr: "Analyzes why retrieval augmentation reduces LM perplexity, ruling out the softmax bottleneck."
---

# On Retrieval Augmentation and the Limitations of Language Model Training

**Source**: [https://aclanthology.org/2024.naacl-short.21/](https://aclanthology.org/2024.naacl-short.21/)

**TLDR**: Analyzes why retrieval augmentation reduces LM perplexity, ruling out the softmax bottleneck.

## Abstract

AbstractAugmenting a language model (LM) with k-nearest neighbors (kNN) retrieval on its training data alone can decrease its perplexity, though the underlying reasons for this remain elusive. In this work, we rule out one previously posited possibility — the “softmax bottleneck.” We then create a new dataset to evaluate LM generalization ability in the setting where training data contains additional information that is not causally relevant. This task is challenging even for GPT-3.5 Turbo. We show that, for both GPT-2 and Mistral 7B, kNN retrieval augmentation consistently improves per formance in this setting. Finally, to make kNN retrieval more accessible, we propose using amulti-layer perceptron model that maps datastore keys to values as a drop-in replacement for traditional retrieval. This reduces storage costsby over 25x.