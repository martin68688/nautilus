---
title: "Pass-Efficient Algorithms for Graph Spectral Clustering (Student Abstract)"
source: "https://www.semanticscholar.org/paper/acbb1adbfecb3b033d14b4253a5b0eda1afbe35f"
categories: ['graph-learning-clustering-multiview-federated', 'brain-inspired-spiking-neural-networks']
tags: ['spectral-clustering', 'graph-algorithms', 'pass-efficient']
venue: "AAAI 2024"
tldr: "Develops pass-efficient algorithms for graph spectral clustering to reduce computational cost."
---

# Pass-Efficient Algorithms for Graph Spectral Clustering (Student Abstract)

**Source**: [https://www.semanticscholar.org/paper/acbb1adbfecb3b033d14b4253a5b0eda1afbe35f](https://www.semanticscholar.org/paper/acbb1adbfecb3b033d14b4253a5b0eda1afbe35f)

**TLDR**: Develops pass-efficient algorithms for graph spectral clustering to reduce computational cost.

## Abstract

Graph spectral clustering is a fundamental technique in data analysis, which utilizes eigenpairs of the Laplacian matrix to partition graph vertices into clusters. However, classical spectral clustering algorithms require eigendecomposition of the Laplacian matrix, which has cubic time complexity. In
this work, we describe pass-efficient spectral clustering algorithms that leverage recent advances in randomized eigendecomposition and the structure of the graph vertex-edge matrix. Furthermore, we derive formulas for their efficient implementation. The resulting algorithms have a linear time complexity with respect to the number of vertices and edges and pass over the graph constant times, making them suitable for processing large graphs stored on slow memory. Experiments validate the accuracy and efficiency of the algorithms.