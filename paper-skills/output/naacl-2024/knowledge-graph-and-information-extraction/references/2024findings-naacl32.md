---
title: "Graph-Induced Syntactic-Semantic Spaces in Transformer-Based Variational AutoEncoders"
source: "https://aclanthology.org/2024.findings-naacl.32/"
categories: ['knowledge-graph-and-information-extraction']
tags: ['variational-autoencoder', 'syntax', 'semantics', 'graph']
venue: "NAACL 2024"
tldr: "Improves VAE performance by separating syntactic and semantic encoding using graph-induced spaces."
---

# Graph-Induced Syntactic-Semantic Spaces in Transformer-Based Variational AutoEncoders

**Source**: [https://aclanthology.org/2024.findings-naacl.32/](https://aclanthology.org/2024.findings-naacl.32/)

**TLDR**: Improves VAE performance by separating syntactic and semantic encoding using graph-induced spaces.

## Abstract

AbstractThe injection of syntactic information in Variational AutoEncoders (VAEs) can result in an overall improvement of performances and generalisation. An effective strategy to achieve such a goal is to separate the encoding of distributional semantic features and syntactic structures into heterogeneous latent spaces via multi-task learning or dual encoder architectures. However, existing works employing such techniques are limited to LSTM-based VAEs. This work investigates latent space separation methods for structural syntactic injection in Transformer-based VAE architectures (i.e., Optimus) through the integration of graph-based models. Our empirical evaluation reveals that the proposed end-to-end VAE architecture can improve theoverall organisation of the latent space, alleviating the information loss occurring in standard VAE setups, and resulting in enhanced performances on language modelling and downstream generation tasks.