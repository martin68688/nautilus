---
title: "Deep Structural Knowledge Exploitation and Synergy for Estimating Node Importance Value on Heterogeneous Information Networks"
source: "https://www.semanticscholar.org/paper/31d86bad9fe4e2cbe1ea20c3e30cbfbe5a601bee"
categories: ['graph-learning-clustering-multiview-federated']
tags: ['node-importance', 'heterogeneous-networks', 'graph-neural-network']
venue: "AAAI 2024"
tldr: "Estimates node importance on heterogeneous information networks by exploiting and synergizing deep structural knowledge."
---

# Deep Structural Knowledge Exploitation and Synergy for Estimating Node Importance Value on Heterogeneous Information Networks

**Source**: [https://www.semanticscholar.org/paper/31d86bad9fe4e2cbe1ea20c3e30cbfbe5a601bee](https://www.semanticscholar.org/paper/31d86bad9fe4e2cbe1ea20c3e30cbfbe5a601bee)

**TLDR**: Estimates node importance on heterogeneous information networks by exploiting and synergizing deep structural knowledge.

## Abstract

The classic problem of node importance estimation has been conventionally studied with homogeneous network topology analysis. To deal with practical network heterogeneity, a few recent methods employ graph neural models to automatically learn diverse sources of information. However, the major concern revolves around that their fully adaptive learning process may lead to insufficient information exploration, thereby formulating the problem as the isolated node value prediction with underperformance and less interpretability. In this work, we propose a novel learning framework namely SKES. Different from previous automatic learning designs, SKES exploits heterogeneous structural knowledge to enrich the informativeness of node representations. Then based on a sufficiently uninformative reference, SKES estimates the importance value for any input node, by quantifying its informativeness disparity against the reference. This establishes an interpretable node importance computation paradigm. Furthermore, SKES dives deep into the understanding that "nodes with similar characteristics are prone to have similar importance values" whilst guaranteeing that such informativeness disparity between any different nodes is orderly reflected by the embedding distance of their associated latent features. Extensive experiments on three widely-evaluated benchmarks demonstrate the performance superiority of SKES over several recent competing methods.