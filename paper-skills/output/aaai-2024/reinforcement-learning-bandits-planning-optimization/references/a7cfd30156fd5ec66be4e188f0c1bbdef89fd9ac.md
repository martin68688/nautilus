---
title: "Parameterized Approximation Algorithms for Sum of Radii Clustering and Variants"
source: "https://www.semanticscholar.org/paper/a7cfd30156fd5ec66be4e188f0c1bbdef89fd9ac"
categories: ['machine-learning-optimization-generalization', 'reinforcement-learning-bandits-planning-optimization', 'sat-satisfiability-solving-and-optimization']
tags: ['clustering', 'approximation-algorithms', 'parameterized-complexity', 'sum-of-radii']
venue: "AAAI 2024"
tldr: "Presents parameterized approximation algorithms for the Sum of Radii clustering problem and its variants."
---

# Parameterized Approximation Algorithms for Sum of Radii Clustering and Variants

**Source**: [https://www.semanticscholar.org/paper/a7cfd30156fd5ec66be4e188f0c1bbdef89fd9ac](https://www.semanticscholar.org/paper/a7cfd30156fd5ec66be4e188f0c1bbdef89fd9ac)

**TLDR**: Presents parameterized approximation algorithms for the Sum of Radii clustering problem and its variants.

## Abstract

Clustering is one of the most fundamental tools in artificial intelligence, machine learning, and data mining. In this paper, we follow one of the recent mainstream topics of clustering, Sum of Radii (SoR), which naturally arises as a balance between the folklore k-center and k-median. SoR aims to determine a set of k balls, each centered at a point in a given dataset, such that their union covers the entire dataset while minimizing the sum of radii of the k balls. 
We propose a general technical framework to overcome the challenge posed by varying radii in SoR, which yields fixed-parameter tractable (fpt) algorithms with respect to k (i.e., whose running time is f(k) ploy(n) for some f). 
Our framework is versatile and obtains fpt approximation algorithms with constant approximation ratios for SoR as well as its variants in general metrics, such as Fair SoR and Matroid SoR, which significantly improve the previous results.