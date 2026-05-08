---
title: "Incorporating Exponential Smoothing into MLP: a Simple but Effective Sequence Model"
source: "https://aclanthology.org/2024.findings-naacl.23/"
categories: ['efficient-transformer-acceleration-techniques']
tags: ['sequence-modeling', 'mlp', 'exponential-smoothing', 'long-range']
venue: "NAACL 2024"
tldr: "Incorporating exponential smoothing into MLPs creates a simple yet effective sequence model."
---

# Incorporating Exponential Smoothing into MLP: a Simple but Effective Sequence Model

**Source**: [https://aclanthology.org/2024.findings-naacl.23/](https://aclanthology.org/2024.findings-naacl.23/)

**TLDR**: Incorporating exponential smoothing into MLPs creates a simple yet effective sequence model.

## Abstract

AbstractModeling long-range dependencies in sequential data is a crucial step in sequence learning. A recently developed model, the Structured State Space (S4), demonstrated significant effectiveness in modeling long-range sequences. However, It is unclear whether the success of S4 can be attributed to its intricate parameterization and HiPPO initialization or simply due to State Space Models (SSMs). To further investigate the potential of the deep SSMs, we start with exponential smoothing (ETS), a simple SSM, and propose a stacked architecture by directly incorporating it into an element-wise MLP. We augment simple ETS with additional parameters and complex field to reduce the inductive bias. Despite increasing less than 1% of parameters of element-wise MLP, our models achieve comparable results to S4 on the LRA benchmark.