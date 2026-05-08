---
title: "Effective Long-Context Scaling of Foundation Models"
source: "https://aclanthology.org/2024.naacl-long.260/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['long-context', 'continual-pretraining', 'llama']
venue: "NAACL 2024"
tldr: "Presents a recipe for continual pretraining to extend LLM context windows to 32k tokens effectively."
---

# Effective Long-Context Scaling of Foundation Models

**Source**: [https://aclanthology.org/2024.naacl-long.260/](https://aclanthology.org/2024.naacl-long.260/)

**TLDR**: Presents a recipe for continual pretraining to extend LLM context windows to 32k tokens effectively.

## Abstract

AbstractWe present an effective recipe to train strong long-context LLMs that are capable of utilizing massive context windows of up to 32,000 tokens. Our models are built through continual pretraining from Llama 2 checkpoints with longer text sequences and on a dataset where long texts are upsampled. We perform extensive evaluation using language modeling, synthetic context probing tasks, and a wide range of downstream benchmarks. Across all evaluations, our models achieve consistent improvements on most regular-context tasks and significant improvements on long-context tasks over Llama 2. Moreover, with a cost-effective instruction tuning procedure that is free of expensive annotation, the presented models can already surpass gpt-3.5-turbo-16k‘s overall performance on long-context benchmarks. Alongside these results, we provide an in-depth analysis on each individual component of our method. We delve into Llama’s position encodings and discuss its key limitation in modeling long data. We examine the impact of various design choices in the pretraining process, including the data mix and the training curriculum of sequence lengths – ablation results suggest that having abundant long texts in the pretrain dataset is not the key to achieving strong performance, and we empirically verify that long context continual pretraining is more efficient and similarly effective compared to pretraining from scratch with long sequences.