---
title: "A Deep Probabilistic Framework for Continuous Time Dynamic Graph Generation"
source: "https://www.semanticscholar.org/paper/5e1ec71f9cc9baaad41c71c4b7ad2bd20fa1c3f7"
categories: ['topological-geometric-data-representation', 'multi-objective-optimization-and-learning']
tags: ['dynamic-graphs', 'generative-models', 'probabilistic-framework']
venue: "AAAI 2024"
tldr: "Presents a deep probabilistic framework for generating continuous-time dynamic graphs with evolving topologies and features."
---

# A Deep Probabilistic Framework for Continuous Time Dynamic Graph Generation

**Source**: [https://www.semanticscholar.org/paper/5e1ec71f9cc9baaad41c71c4b7ad2bd20fa1c3f7](https://www.semanticscholar.org/paper/5e1ec71f9cc9baaad41c71c4b7ad2bd20fa1c3f7)

**TLDR**: Presents a deep probabilistic framework for generating continuous-time dynamic graphs with evolving topologies and features.

## Abstract

Recent advancements in graph representation learning have shifted attention towards dynamic graphs, which exhibit evolving topologies and features over time. The increased use of such graphs creates a paramount need for generative models suitable for applications such as data augmentation, obfuscation, and anomaly detection. However, there are few generative techniques that handle continuously changing temporal graph data; existing work largely relies on augmenting static graphs with additional temporal information to model dynamic interactions between nodes. In this work, we propose a fundamentally different approach: We instead directly model interactions as a joint probability of an edge forming between two nodes at a given time. This allows us to autoregressively generate new synthetic dynamic graphs in a largely assumption free, scalable, and inductive manner. We formalize this approach as DG-Gen, a generative framework for continuous time dynamic graphs, and demonstrate its effectiveness over five datasets. Our experiments demonstrate that DG-Gen not only generates higher fidelity graphs compared to traditional methods but also significantly advances link prediction tasks.