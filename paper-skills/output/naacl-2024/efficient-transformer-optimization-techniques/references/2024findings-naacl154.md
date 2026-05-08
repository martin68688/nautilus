---
title: "A Lightweight Mixture-of-Experts Neural Machine Translation Model with Stage-wise Training Strategy"
source: "https://aclanthology.org/2024.findings-naacl.154/"
categories: ['efficient-transformer-optimization-techniques', 'speech-language-processing-multilingual']
tags: ['mixture-of-experts', 'neural-machine-translation', 'model-efficiency']
venue: "NAACL 2024"
tldr: "Presents a lightweight mixture-of-experts NMT model with a stage-wise training strategy to handle language heterogeneity."
---

# A Lightweight Mixture-of-Experts Neural Machine Translation Model with Stage-wise Training Strategy

**Source**: [https://aclanthology.org/2024.findings-naacl.154/](https://aclanthology.org/2024.findings-naacl.154/)

**TLDR**: Presents a lightweight mixture-of-experts NMT model with a stage-wise training strategy to handle language heterogeneity.

## Abstract

AbstractDealing with language heterogeneity has always been one of the challenges in neural machine translation (NMT).The idea of using mixture-of-experts (MoE) naturally excels in addressing this issue by employing different experts to take responsibility for different problems.However, the parameter-inefficiency problem in MoE results in less performance improvement when boosting the number of parameters.Moreover, most of the MoE models are suffering from the training instability problem.This paper proposes MoA (Mixture-of-Adapters), a lightweight MoE-based NMT model that is trained via an elaborately designed stage-wise training strategy.With the standard Transformer as the backbone model, we introduce lightweight adapters as experts for easy expansion.To improve the parameter efficiency, we explicitly model and distill the language heterogeneity into the gating network with clustering.After freezing the gating network, we adopt the Gumbel-Max sampling as the routing scheme when training experts to balance the knowledge of generalization and specialization while preventing expert over-fitting.Empirical results show that MoA achieves stable improvements in different translation tasks by introducing much fewer extra parameters compared to other MoE baselines.Additionally, the performance evaluations on a multi-domain translation task illustrate the effectiveness of our training strategy.