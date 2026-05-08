---
title: "Efficient Self-Supervised Video Hashing with Selective State Spaces"
source: "https://www.semanticscholar.org/paper/da45c4d7ab6ab629acf6274df7635d7ea7dbf2ef"
categories: ['llm-safety-adversarial-robustness', 'multimodal-time-series-foundation-models']
tags: ['video', 'hashing', 'self-supervised', 'selective-state-spaces']
venue: "AAAI 2024"
tldr: "Improves self-supervised video hashing efficiency with selective state spaces inspired by Mamba."
---

# Efficient Self-Supervised Video Hashing with Selective State Spaces

**Source**: [https://www.semanticscholar.org/paper/da45c4d7ab6ab629acf6274df7635d7ea7dbf2ef](https://www.semanticscholar.org/paper/da45c4d7ab6ab629acf6274df7635d7ea7dbf2ef)

**TLDR**: Improves self-supervised video hashing efficiency with selective state spaces inspired by Mamba.

## Abstract

Self-supervised video hashing (SSVH) is a practical task in video indexing and retrieval. Although Transformers are predominant in SSVH for their impressive temporal modeling capabilities, they often suffer from computational and memory inefficiencies. Drawing inspiration from Mamba, an advanced state-space model, we explore its potential in SSVH to achieve a better balance between efficacy and efficiency. We introduce S5VH, a Mamba-based video hashing model with an improved self-supervised learning paradigm. Specifically, we design bidirectional Mamba layers for both the encoder and decoder, which are effective and efficient in capturing temporal relationships thanks to the data-dependent selective scanning mechanism with linear complexity. In our learning strategy, we transform global semantics in the feature space into semantically consistent and discriminative hash centers, followed by a center alignment loss as a global learning signal. Our self-local-global (SLG) paradigm significantly improves learning efficiency, leading to faster and better convergence. Extensive experiments demonstrate S5VH's improvements over state-of-the-art methods, superior transferability, and scalable advantages in inference efficiency.