---
title: "Attention Alignment and Flexible Positional Embeddings Improve Transformer Length Extrapolation"
source: "https://aclanthology.org/2024.findings-naacl.10/"
categories: ['efficient-transformer-optimization-techniques', 'knowledge-graph-and-information-extraction']
tags: ['transformer', 'length-extrapolation', 'positional-embedding', 'attention']
venue: "NAACL 2024"
tldr: "Improving Transformer length extrapolation via attention alignment and flexible positional embeddings."
---

# Attention Alignment and Flexible Positional Embeddings Improve Transformer Length Extrapolation

**Source**: [https://aclanthology.org/2024.findings-naacl.10/](https://aclanthology.org/2024.findings-naacl.10/)

**TLDR**: Improving Transformer length extrapolation via attention alignment and flexible positional embeddings.

## Abstract

AbstractAn ideal length-extrapolatable Transformer language model can handle sequences longer than the training length without any fine-tuning. Such long-context utilization capability relies heavily on a flexible positional embedding design. Upon investigating the flexibility of existing large pre-trained Transformer language models, we find that the T5 family deserves a closer look, as its positional embeddings capture rich and flexible attention patterns. However, T5 suffers from the dispersed attention issue: the longer the input sequence, the flatter the attention distribution. To alleviate the issue, we propose two attention alignment strategies via temperature scaling. Our findings show improvement on the long-context utilization capability of T5 on language modeling, retrieval, multi-document question answering, and code completion tasks without any fine-tuning. This suggests that a flexible positional embedding design and attention alignment can go a long way toward Transformer length extrapolation. The code is released at: https://github.com/chijames/T5-Attention-Alignment