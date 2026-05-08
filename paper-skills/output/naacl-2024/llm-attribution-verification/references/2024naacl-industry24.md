---
title: "Scaling Up Authorship Attribution"
source: "https://aclanthology.org/2024.naacl-industry.24/"
categories: ['llm-attribution-verification']
tags: ['authorship-attribution', 'scaling', 'infrastructure']
venue: "NAACL 2024"
tldr: "Describing a scalable system for authorship attribution under specific runtime and use-case constraints."
---

# Scaling Up Authorship Attribution

**Source**: [https://aclanthology.org/2024.naacl-industry.24/](https://aclanthology.org/2024.naacl-industry.24/)

**TLDR**: Describing a scalable system for authorship attribution under specific runtime and use-case constraints.

## Abstract

AbstractWe describe our system for authorship attribution in the IARPA HIATUS program. We describe the model and compute infrastructure developed to satisfy the set of technical constraints imposed by IARPA, including runtime limits as well as other constraints related to the ultimate use case. One use-case constraint concerns the explainability of the features used in the system. For this reason, we integrate features from frame semantic parsing, as they are both interpretable and difficult for adversaries to evade. One trade-off with using such features, however, is that more sophisticated feature representations require more complicated architectures, which limit usefulness in time-sensitive and constrained compute environments. We propose an approach to increase the efficiency of frame semantic parsing through an analysis of parallelization and beam search sizes. Our approach results in a system that is approximately 8.37x faster than the base system with a minimal effect on accuracy.