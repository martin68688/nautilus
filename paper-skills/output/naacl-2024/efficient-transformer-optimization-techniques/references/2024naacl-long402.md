---
title: "The Impact of Depth on Compositional Generalization in Transformer Language Models"
source: "https://aclanthology.org/2024.naacl-long.402/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques', 'metaphor-analysis-political-framing']
tags: ['compositional-generalization', 'transformer', 'depth']
venue: "NAACL 2024"
tldr: "Investigates how transformer depth impacts compositional generalization, testing the hypothesis that deeper layers promote it."
---

# The Impact of Depth on Compositional Generalization in Transformer Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.402/](https://aclanthology.org/2024.naacl-long.402/)

**TLDR**: Investigates how transformer depth impacts compositional generalization, testing the hypothesis that deeper layers promote it.

## Abstract

AbstractTo process novel sentences, language models (LMs) must generalize compositionally—combine familiar elements in new ways. What aspects of a model’s structure promote compositional generalization? Focusing on transformers, we test the hypothesis, motivated by theoretical and empirical work, that deeper transformers generalize more compositionally. Simply adding layers increases the total number of parameters; to address this confound between depth and size, we construct three classes of models which trade off depth for width such that the total number of parameters is kept constant (41M, 134M and 374M parameters). We pretrain all models as LMs and fine-tune them on tasks that test for compositional generalization. We report three main conclusions: (1) after fine-tuning, deeper models generalize more compositionally than shallower models do, but the benefit of additional layers diminishes rapidly; (2) within each family, deeper models show better language modeling performance, but returns are similarly diminishing; (3) the benefits of depth for compositional generalization cannot be attributed solely to better performance on language modeling. Because model latency is approximately linear in the number of layers, these results lead us to the recommendation that, with a given total parameter budget, transformers can be made shallower than is typical without sacrificing performance.