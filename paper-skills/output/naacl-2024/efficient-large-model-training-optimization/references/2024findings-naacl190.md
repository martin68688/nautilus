---
title: "Retrieving Examples from Memory for Retrieval Augmented Neural Machine Translation: A Systematic Comparison"
source: "https://aclanthology.org/2024.findings-naacl.190/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['retrieval-augmentation', 'memory', 'translation']
venue: "NAACL 2024"
tldr: "Systematically compares retrieval methods for RAMT, focusing on the upstream retrieval step."
---

# Retrieving Examples from Memory for Retrieval Augmented Neural Machine Translation: A Systematic Comparison

**Source**: [https://aclanthology.org/2024.findings-naacl.190/](https://aclanthology.org/2024.findings-naacl.190/)

**TLDR**: Systematically compares retrieval methods for RAMT, focusing on the upstream retrieval step.

## Abstract

AbstractRetrieval-Augmented Neural Machine Translation (RAMT) architectures retrieve examples from memory to guide the generation process. While most works in this trend explore new ways to exploit the retrieved examples, the upstream retrieval step is mostly unexplored. In this paper, we study the effect of varying retrieval methods for several translation architectures to better understand the interplay between these two processes.We conduct experiments in two language pairs in a multi-domain setting and consider several downstream architectures based on a standard autoregressive model, an edit-based model, and a large language model with in-context learning. Our experiments show that the choice of the retrieval technique impacts the translation scores, with variance across architectures. We also discuss the effects of increasing the number and diversity of examples, which are mostly positive across the board.