---
title: "Rehearsal-Free Modular and Compositional Continual Learning for Language Models"
source: "https://aclanthology.org/2024.naacl-short.39/"
categories: ['continual-learning-memory-transfer-llms', 'efficient-large-model-training-optimization']
tags: ['continual-learning', 'modular', 'compositional', 'rehearsal-free']
venue: "NAACL 2024"
tldr: "Proposes a rehearsal-free, modular, and compositional continual learning method for language models."
---

# Rehearsal-Free Modular and Compositional Continual Learning for Language Models

**Source**: [https://aclanthology.org/2024.naacl-short.39/](https://aclanthology.org/2024.naacl-short.39/)

**TLDR**: Proposes a rehearsal-free, modular, and compositional continual learning method for language models.

## Abstract

AbstractContinual learning aims at incrementally acquiring new knowledge while not forgetting existing knowledge. To overcome catastrophic forgetting, methods are either rehearsal-based, i.e., store data examples from previous tasks for data replay, or isolate parameters dedicated to each task. However, rehearsal-based methods raise privacy and memory issues, and parameter-isolation continual learning does not consider interaction between tasks, thus hindering knowledge transfer. In this work, we propose MoCL, a rehearsal-free **Mo**dular and **C**ompositional Continual **L**earning framework which continually adds new modules to language models and composes them with existing modules. Experiments on various benchmarks show that MoCL outperforms state of the art and effectively facilitates knowledge transfer.