---
title: "ARM: Alignment with Residual Energy-Based Model"
source: "https://aclanthology.org/2024.naacl-long.455/"
categories: ['efficient-large-model-training-optimization', 'llm-alignment-safety-detoxification']
tags: ['alignment', 'energy-based-model', 'rlhf', 'residual-model']
venue: "NAACL 2024"
tldr: "Introduces an alignment method using a residual energy-based model to improve LLM responses according to human preferences."
---

# ARM: Alignment with Residual Energy-Based Model

**Source**: [https://aclanthology.org/2024.naacl-long.455/](https://aclanthology.org/2024.naacl-long.455/)

**TLDR**: Introduces an alignment method using a residual energy-based model to improve LLM responses according to human preferences.

## Abstract

AbstractWhile large language models (LLMs) trained with large-scale unsupervised learning acquire a wide variety of world knowledge and skills, its behavior does not necessarily align with human preferences. RLHF methods achieve successes in aligning LLM responses with human preferences and improving the controllability of LLM behavior with human instruction. However, RLHF methods are considerably complicated to implement, computationally expensive to train, and notoriously tricky to tune. In this work, we propose Alignment with Residual Energy-Based Model (ARM), as a simple and flexible alternative to RLHF methods. Our method is driven by an observation that we can learn an aligned policy by minimizing a forward Kullback–Leibler (KL) divergence from a target policy (in the form of a residual energy-based model) to a parameteric policy (LLM), instead of a reverse KL as in RLHF methods. With samples from the energy-based target policy, we can leverage the power of DPO (or other offline methods) to learn an aligned policy efficiently. ARM is simple to implement and applicable in various data settings. Our extensive experiments demonstrate its strong performance across multiple datasets, compared to strong baselines like PPO, DPO.