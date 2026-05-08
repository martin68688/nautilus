---
title: "Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation"
source: "https://aclanthology.org/2024.naacl-long.214/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['efficient-training', 'auto-regressive', 'event-graph']
venue: "NAACL 2024"
tldr: "Proposes a set-aligning framework for auto-regressive generation of event temporal graphs."
---

# Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation

**Source**: [https://aclanthology.org/2024.naacl-long.214/](https://aclanthology.org/2024.naacl-long.214/)

**TLDR**: Proposes a set-aligning framework for auto-regressive generation of event temporal graphs.

## Abstract

AbstractEvent temporal graphs have been shown as convenient and effective representations of complex temporal relations between events in text. Recent studies, which employ pre-trained language models to auto-regressively generate linearised graphs for constructing event temporal graphs, have shown promising results. However, these methods have often led to suboptimal graph generation as the linearised graphs exhibit set characteristics which are instead treated sequentially by language models. This discrepancy stems from the conventional text generation objectives, leading to erroneous penalisation of correct predictions caused by the misalignment of elements in target sequences. To address these challenges, we reframe the task as a conditional set generation problem, proposing a Set-aligning Framework tailored for the effective utilisation of Large Language Models (LLMs). The framework incorporates data augmentations and set-property regularisations designed to alleviate text generation loss penalties associated with the linearised graph edge sequences, thus encouraging the generation of more relation edges. Experimental results show that our framework surpasses existing baselines for event temporal graph generation. Furthermore, under zero-shot settings, the structural knowledge introduced through our framework notably improves model generalisation, particularly when the training examples available are limited.