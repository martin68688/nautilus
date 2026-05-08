---
title: "Towards a Theoretical Understanding of Why Local Search Works for Clustering with Fair-Center Representation"
source: "https://www.semanticscholar.org/paper/255d2beef4dec77bce9f5c4ae1be0aae5c9a2261"
categories: ['reasoning-learning-optimization-in-llms', 'fairness-constrained-clustering-approximation']
tags: ['fair-clustering', 'approximation-algorithms', 'local-search']
venue: "AAAI 2024"
tldr: "Provides a theoretical understanding of why local search algorithms are effective for fair k-median clustering problems with representation constraints."
---

# Towards a Theoretical Understanding of Why Local Search Works for Clustering with Fair-Center Representation

**Source**: [https://www.semanticscholar.org/paper/255d2beef4dec77bce9f5c4ae1be0aae5c9a2261](https://www.semanticscholar.org/paper/255d2beef4dec77bce9f5c4ae1be0aae5c9a2261)

**TLDR**: Provides a theoretical understanding of why local search algorithms are effective for fair k-median clustering problems with representation constraints.

## Abstract

The representative k-median problem generalizes the classical clustering formulations in that it partitions the data points into several disjoint demographic groups and poses a lower-bound constraint on the number of opened facilities from each group, such that all the groups are fairly represented by the opened facilities. Due to its simplicity, the local-search heuristic that optimizes an initial solution by iteratively swapping at most a constant number of closed facilities for the same number of opened ones (denoted by the O(1)-swap heuristic) has been frequently used in the representative k-median problem. Unfortunately, despite its good performance exhibited in experiments, whether the O(1)-swap heuristic has provable approximation guarantees for the case where the number of groups is more than 2 remains an open question for a long time. As an answer to this question, we show that the O(1)-swap heuristic
(1) is guaranteed to yield a constant-factor approximation solution if the number of groups is a constant, and
(2) has an unbounded approximation ratio otherwise.
Our main technical contribution is a new approach for theoretically analyzing local-search heuristics, which derives the approximation ratio of the O(1)-swap heuristic via linearly combining the increased clustering costs induced by a set of hierarchically organized swaps.