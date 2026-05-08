---
title: "Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations"
source: "https://aclanthology.org/2024.naacl-long.89/"
categories: ['large-language-model-evaluation-augmentation', 'contrastive-and-generative-representation-learning']
tags: ['contrastive-learning', 'semantic-representation', 'sub-sentence']
venue: "NAACL 2024"
tldr: "Introduces a contrastively-learned sub-sentence encoder for fine-grained semantic representations."
---

# Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations

**Source**: [https://aclanthology.org/2024.naacl-long.89/](https://aclanthology.org/2024.naacl-long.89/)

**TLDR**: Introduces a contrastively-learned sub-sentence encoder for fine-grained semantic representations.

## Abstract

AbstractWe introduce sub-sentence encoder, a contrastively-learned contextual embedding model for fine-grained semantic representation of text. In contrast to the standard practice with sentence embeddings, where the meaning of an entire sequence of text is encoded into a fixed-length vector, the sub-sentence encoder learns to produce distinct contextual embeddings corresponding to different atomic propositions, i.e. atomic units of meaning expressed within a text sequence. The sub-sentence embeddings are contrastively learned to recognize (inferred) semantic equivalence between propositions across different text sequences. Our experiments show the effectiveness of sub-sentence encoders in applications, such as retrieving supporting facts for fine-grained text attribution or recognizing the conditional semantic similarity between texts. In practice, we demonstrate that sub-sentence encoders keep the same level of inference cost and space complexity compared to sentence encoders.