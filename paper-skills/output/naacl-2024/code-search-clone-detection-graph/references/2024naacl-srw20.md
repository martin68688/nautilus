---
title: "Source Code is a Graph, Not a Sequence: A Cross-Lingual Perspective on Code Clone Detection"
source: "https://aclanthology.org/2024.naacl-srw.20/"
categories: ['code-search-clone-detection-graph']
tags: ['code-clone', 'graph-representation', 'cross-lingual']
venue: "NAACL 2024"
tldr: "Argues for graph-based over sequence-based methods for cross-lingual code clone detection."
---

# Source Code is a Graph, Not a Sequence: A Cross-Lingual Perspective on Code Clone Detection

**Source**: [https://aclanthology.org/2024.naacl-srw.20/](https://aclanthology.org/2024.naacl-srw.20/)

**TLDR**: Argues for graph-based over sequence-based methods for cross-lingual code clone detection.

## Abstract

AbstractCode clone detection is challenging, as sourcecode can be written in different languages, do-mains, and styles. In this paper, we arguethat source code is inherently a graph, not asequence, and that graph-based methods aremore suitable for code clone detection thansequence-based methods. We compare the per-formance of two state-of-the-art models: Code-BERT (Feng et al., 2020), a sequence-basedmodel, and CodeGraph (Yu et al., 2023), agraph-based model, on two benchmark data-sets: BCB (Svajlenko et al., 2014) and PoolC(PoolC, no date). We show that CodeGraphoutperforms CodeBERT on both data-sets, es-pecially on cross-lingual code clones. To thebest of our knowledge, this is the first work todemonstrate the cross-lingual code clone detec-tion showing superiority on graph-based meth-ods over sequence-based methods