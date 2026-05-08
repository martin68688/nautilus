---
title: "mOthello: When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models?"
source: "https://aclanthology.org/2024.findings-naacl.103/"
categories: ['speech-language-processing-multilingual']
tags: ['multilingual', 'representation-alignment', 'cross-lingual-transfer']
venue: "NAACL 2024"
tldr: "Investigates when cross-lingual representation alignment and transfer emerge in multilingual models using a probing task."
---

# mOthello: When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models?

**Source**: [https://aclanthology.org/2024.findings-naacl.103/](https://aclanthology.org/2024.findings-naacl.103/)

**TLDR**: Investigates when cross-lingual representation alignment and transfer emerge in multilingual models using a probing task.

## Abstract

AbstractMany pretrained multilingual models exhibit cross-lingual transfer ability, which is often attributed to a learned language-neutral representation during pretraining. However, it remains unclear what factors contribute to the learning of a language-neutral representation, and whether the learned language-neutral representation suffices to facilitate cross-lingual transfer. We propose a synthetic task, Multilingual Othello (mOthello), as a testbed to delve into these two questions. We find that: (1) models trained with naive multilingual pretraining fail to learn a language-neutral representation across all input languages; (2) the introduction of “anchor tokens” (i.e., lexical items that are identical across languages) helps cross-lingual representation alignment; and (3) the learning of a language-neutral representation alone is not sufficient to facilitate cross-lingual transfer. Based on our findings, we propose a novel approach – multilingual pretraining with unified output space – that both induces the learning of language-neutral representation and facilitates cross-lingual transfer.