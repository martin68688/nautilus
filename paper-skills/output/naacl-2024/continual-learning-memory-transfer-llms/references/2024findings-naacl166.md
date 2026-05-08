---
title: "Q-Tuning: Queue-based Prompt Tuning for Lifelong Few-shot Language Learning"
source: "https://aclanthology.org/2024.findings-naacl.166/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'continual-learning-memory-transfer-llms']
tags: ['continual-learning', 'prompt-tuning', 'lifelong-learning', 'few-shot']
venue: "NAACL 2024"
tldr: "Introduces queue-based prompt tuning (Q-tuning) for lifelong few-shot language learning across sequential tasks."
---

# Q-Tuning: Queue-based Prompt Tuning for Lifelong Few-shot Language Learning

**Source**: [https://aclanthology.org/2024.findings-naacl.166/](https://aclanthology.org/2024.findings-naacl.166/)

**TLDR**: Introduces queue-based prompt tuning (Q-tuning) for lifelong few-shot language learning across sequential tasks.

## Abstract

AbstractThis paper introduces Q-tuning, a novel approach for continual prompt tuning that enables the lifelong learning of a pre-trained language model. When learning a new task, Q-tuning trains a task-specific prompt by adding it to a prompt queue consisting of the prompts from older tasks. To better transfer the knowledge of old tasks, we design an adaptive knowledge aggregation technique that reweighs previous prompts in the queue with a learnable low-rank matrix. Once the prompt queue reaches its maximum capacity, we leverage a PCA-based eviction rule to reduce the queue’s size, allowing the newly trained prompt to be added while preserving the primary knowledge of old tasks. In order to mitigate the accumulation of information loss caused by the eviction, we additionally propose a globally shared prefix prompt and a memory retention regularization based on information theory. Extensive experiments demonstrate that our approach outperforms the state-of-the-art methods substantially on continual prompt tuning benchmarks. Moreover, our approach enables lifelong learning on linearly growing task sequences while requiring constant complexity for training and inference.