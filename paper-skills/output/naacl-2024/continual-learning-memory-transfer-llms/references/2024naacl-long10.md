---
title: "Self-generated Replay Memories for Continual Neural Machine Translation"
source: "https://aclanthology.org/2024.naacl-long.10/"
categories: ['continual-learning-memory-transfer-llms']
tags: ['continual-learning', 'machine-translation', 'replay', 'catastrophic-forgetting']
venue: "NAACL 2024"
tldr: "Mitigating catastrophic forgetting in continual neural machine translation using self-generated replay memories."
---

# Self-generated Replay Memories for Continual Neural Machine Translation

**Source**: [https://aclanthology.org/2024.naacl-long.10/](https://aclanthology.org/2024.naacl-long.10/)

**TLDR**: Mitigating catastrophic forgetting in continual neural machine translation using self-generated replay memories.

## Abstract

AbstractModern Neural Machine Translation systems exhibit strong performance in several different languages and are constantly improving. Their ability to learn continuously is, however, still severely limited by the catastrophic forgetting issue. In this work, we leverage a key property of encoder-decoder Transformers, i.e. their generative ability, to propose a novel approach to continually learning Neural Machine Translation systems. We show how this can effectively learn on a stream of experiences comprising different languages, by leveraging a replay memory populated by using the model itself as a generator of parallel sentences. We empirically demonstrate that our approach can counteract catastrophic forgetting without requiring explicit memorization of training data. Code will be publicly available upon publication.