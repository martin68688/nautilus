---
title: "Retrieval-Augmented Primitive Representations for Compositional Zero-Shot Learning"
source: "https://www.semanticscholar.org/paper/b2cb7d8df63d395593fccd6799db3efab7efbeea"
categories: ['ai-agents-psychology-personality-simulation']
tags: ['zero-shot-learning', 'compositional', 'retrieval-augmented']
venue: "AAAI 2024"
tldr: "Uses retrieval-augmented primitive representations for compositional zero-shot learning of unseen attribute-object pairs."
---

# Retrieval-Augmented Primitive Representations for Compositional Zero-Shot Learning

**Source**: [https://www.semanticscholar.org/paper/b2cb7d8df63d395593fccd6799db3efab7efbeea](https://www.semanticscholar.org/paper/b2cb7d8df63d395593fccd6799db3efab7efbeea)

**TLDR**: Uses retrieval-augmented primitive representations for compositional zero-shot learning of unseen attribute-object pairs.

## Abstract

Compositional zero-shot learning (CZSL) aims to recognize unseen attribute-object compositions by learning from seen compositions. Composing the learned knowledge of seen primitives, i.e., attributes or objects, into novel compositions is critical for CZSL. In this work, we propose to explicitly retrieve knowledge of seen primitives for compositional zero-shot learning. We present a retrieval-augmented method, which augments standard multi-path classification methods with two retrieval modules. Specifically, we construct two databases storing the attribute and object representations of training images, respectively. For an input training/testing image, we use two retrieval modules to retrieve representations of training images with the same attribute and object, respectively. The primitive representations of the input image are augmented by using the retrieved representations, for composition recognition. By referencing semantically similar images, the proposed method is capable of recalling knowledge of seen primitives for compositional generalization. Experiments on three widely-used datasets show the effectiveness of the proposed method.