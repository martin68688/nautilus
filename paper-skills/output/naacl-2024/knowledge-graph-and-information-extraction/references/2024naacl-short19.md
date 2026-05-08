---
title: "Separately Parameterizing Singleton Detection Improves End-to-end Neural Coreference Resolution"
source: "https://aclanthology.org/2024.naacl-short.19/"
categories: ['knowledge-graph-and-information-extraction', 'dependency-tree-sampling-and-annotation']
tags: ['coreference-resolution', 'singleton-detection', 'neural-models']
venue: "NAACL 2024"
tldr: "Improves neural coreference resolution by separately parameterizing the detection of singleton mentions from antecedent linking."
---

# Separately Parameterizing Singleton Detection Improves End-to-end Neural Coreference Resolution

**Source**: [https://aclanthology.org/2024.naacl-short.19/](https://aclanthology.org/2024.naacl-short.19/)

**TLDR**: Improves neural coreference resolution by separately parameterizing the detection of singleton mentions from antecedent linking.

## Abstract

AbstractCurrent end-to-end coreference resolution models combine detection of singleton mentions and antecedent linking into a single step. In contrast, singleton detection was often treated as a separate step in the pre-neural era. In this work, we show that separately parameterizing these two sub-tasks also benefits end-to-end neural coreference systems. Specifically, we add a singleton detector to the coarse-to-fine (C2F) coreference model, and design an anaphoricity-aware span embedding and singleton detection loss. Our method significantly improves model performance on OntoNotes and four additional datasets.