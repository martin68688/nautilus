---
title: "Advancing Regular Language Reasoning in Linear Recurrent Neural Networks"
source: "https://aclanthology.org/2024.naacl-short.4/"
categories: ['efficient-transformer-acceleration-techniques', 'neural-logical-reasoning-interpretability']
tags: ['recurrent-neural-networks', 'formal-languages', 'reasoning', 'linear-rnn']
venue: "NAACL 2024"
tldr: "Investigating the capability of linear recurrent neural networks to learn and reason with regular languages."
---

# Advancing Regular Language Reasoning in Linear Recurrent Neural Networks

**Source**: [https://aclanthology.org/2024.naacl-short.4/](https://aclanthology.org/2024.naacl-short.4/)

**TLDR**: Investigating the capability of linear recurrent neural networks to learn and reason with regular languages.

## Abstract

AbstractIn recent studies, linear recurrent neural networks (LRNNs) have achieved Transformer-level performance in natural language and long-range modeling, while offering rapid parallel training and constant inference cost. With the resurgence of interest in LRNNs, we study whether they can learn the hidden rules in training sequences, such as the grammatical structures of regular language. We theoretically analyze some existing LRNNs and discover their limitations in modeling regular language. Motivated by this analysis, we propose a new LRNN equipped with a block-diagonal and input-dependent transition matrix. Experiments suggest that the proposed model is the only LRNN capable of performing length extrapolation on regular language tasks such as Sum, Even Pair, and Modular Arithmetic. The code is released at https://github.com/tinghanf/RegluarLRNN.