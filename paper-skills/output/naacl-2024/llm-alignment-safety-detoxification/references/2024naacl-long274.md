---
title: "Anisotropy is Not Inherent to Transformers"
source: "https://aclanthology.org/2024.naacl-long.274/"
categories: ['efficient-large-model-training-optimization', 'llm-alignment-safety-detoxification']
tags: ['embedding-isotropy', 'representation-degradation', 'transformers']
venue: "NAACL 2024"
tldr: "Argues that anisotropy in Transformer embeddings is not inherent and can be addressed with proper training techniques."
---

# Anisotropy is Not Inherent to Transformers

**Source**: [https://aclanthology.org/2024.naacl-long.274/](https://aclanthology.org/2024.naacl-long.274/)

**TLDR**: Argues that anisotropy in Transformer embeddings is not inherent and can be addressed with proper training techniques.

## Abstract

AbstractIsotropy is the property that embeddings are uniformly distributed around the origin. Previous work has shown that Transformer embedding spaces are anisotropic, which is called the representation degradation problem. This degradation has been assumed to be inherent to the standard language modeling tasks and to apply to all Transformer models regardless of their architecture. In this work we identify a set of Transformer models with isotropic embedding spaces, the large Pythia models. We examine the isotropy of Pythia models and explore how isotropy and anisotropy develop as a model is trained. We find that anisotropic models do not develop as previously theorized, using our own analysis to show that the large Pythia models optimize their final Layer Norm for isotropy, and provide reasoning why previous theoretical justifications for anisotropy were insufficient. The identification of a set of isotropic Transformer models calls previous assumptions into question, provides a set of models to contrast existing analysis, and should lead to deeper insight into isotropy.