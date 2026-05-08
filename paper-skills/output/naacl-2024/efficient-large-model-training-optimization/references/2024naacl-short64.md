---
title: "MEMORY-VQ: Compression for Tractable Internet-Scale Memory"
source: "https://aclanthology.org/2024.naacl-short.64/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['retrieval-augmentation', 'compression', 'memory']
venue: "NAACL 2024"
tldr: "Proposes a compression method for tractable internet-scale memory in retrieval-augmented LMs."
---

# MEMORY-VQ: Compression for Tractable Internet-Scale Memory

**Source**: [https://aclanthology.org/2024.naacl-short.64/](https://aclanthology.org/2024.naacl-short.64/)

**TLDR**: Proposes a compression method for tractable internet-scale memory in retrieval-augmented LMs.

## Abstract

AbstractRetrieval augmentation is a powerful but expensive method to make language models more knowledgeable about the world. Memory-based methods like LUMEN (de Jong et al., 2023a) pre-compute token representations for retrieved passages to drastically speed up inference. However, memory also leads to much greater storage requirements from storing pre-computed representations. We propose MEMORY-VQ, a new method to reduce storage requirements of memory-augmented models without sacrificing performance. Our method uses a vector quantization variational autoencoder (VQ-VAE) to compress token representations. We apply MEMORY-VQ to the LUMEN model to obtain LUMEN-VQ, a memory model that achieves a 16x compression rate with comparable performance on the KILT benchmark. LUMEN-VQ enables practical retrieval augmentation even for extremely large retrieval corpora.