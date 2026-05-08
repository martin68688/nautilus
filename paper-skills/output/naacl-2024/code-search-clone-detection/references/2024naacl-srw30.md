---
title: "HybridBERT - Making BERT Pretraining More Efficient Through Hybrid Mixture of Attention Mechanisms"
source: "https://aclanthology.org/2024.naacl-srw.30/"
categories: ['code-search-clone-detection', 'efficient-large-model-training-optimization']
tags: ['efficient-pretraining', 'hybrid-attention']
venue: "NAACL 2024"
tldr: "HybridBERT makes BERT pretraining more efficient with a hybrid mixture of attention mechanisms."
---

# HybridBERT - Making BERT Pretraining More Efficient Through Hybrid Mixture of Attention Mechanisms

**Source**: [https://aclanthology.org/2024.naacl-srw.30/](https://aclanthology.org/2024.naacl-srw.30/)

**TLDR**: HybridBERT makes BERT pretraining more efficient with a hybrid mixture of attention mechanisms.

## Abstract

AbstractPretrained transformer-based language models have produced state-of-the-art performance in most natural language understanding tasks. These models undergo two stages of training: pretraining on a huge corpus of data and fine-tuning on a specific downstream task. The pretraining phase is extremely compute-intensive and requires several high-performance computing devices like GPUs and several days or even months of training, but it is crucial for the model to capture global knowledge and also has a significant impact on the fine-tuning task. This is a major roadblock for researchers without access to sophisticated computing resources. To overcome this challenge, we propose two novel hybrid architectures called HybridBERT (HBERT), which combine self-attention and additive attention mechanisms together with sub-layer normalization. We introduce a computing budget to the pretraining phase, limiting the training time and usage to a single GPU. We show that HBERT attains twice the pretraining accuracy of a vanilla-BERT baseline. We also evaluate our proposed models on two downstream tasks, where we outperform BERT-base while accelerating inference. Moreover, we study the effect of weight initialization with a limited pretraining budget. The code and models are publicly available at: www.github.com/gokulsg/HBERT/.