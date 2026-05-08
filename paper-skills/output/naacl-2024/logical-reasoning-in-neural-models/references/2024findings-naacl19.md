---
title: "GOLD: Geometry Problem Solver with Natural Language Description"
source: "https://aclanthology.org/2024.findings-naacl.19/"
categories: ['continual-learning-memory-transfer-llms', 'logical-reasoning-in-neural-models']
tags: ['geometry', 'math-problem-solving', 'multimodal']
venue: "NAACL 2024"
tldr: "Introduces GOLD, a geometry problem solver that uses natural language descriptions and diagram interpretation."
---

# GOLD: Geometry Problem Solver with Natural Language Description

**Source**: [https://aclanthology.org/2024.findings-naacl.19/](https://aclanthology.org/2024.findings-naacl.19/)

**TLDR**: Introduces GOLD, a geometry problem solver that uses natural language descriptions and diagram interpretation.

## Abstract

AbstractAddressing the challenge of automated geometry math problem-solving in artificial intelligence (AI) involves understanding multi-modal information and mathematics. blackCurrent methods struggle with accurately interpreting geometry diagrams, which hinders effective problem-solving. To tackle this issue, we present the Geometry problem sOlver with natural Language Description (GOLD) model. GOLD enhances the extraction of geometric relations by separately processing symbols and geometric primitives within the diagram. Subsequently, it converts the extracted relations into natural language descriptions, efficiently utilizing large language models to solve geometry math problems. Experiments show that the GOLD model outperforms the Geoformer model, the previous best method on the UniGeo dataset, by achieving accuracy improvements of 12.7% and 42.1% in calculation and proving subsets. Additionally, it surpasses the former best model on the PGPS9K and Geometry3K datasets, PGPSNet, by obtaining accuracy enhancements of 1.8% and 3.2%, respectively.