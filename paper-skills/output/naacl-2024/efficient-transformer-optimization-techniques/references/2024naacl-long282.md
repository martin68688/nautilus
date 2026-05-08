---
title: "AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning"
source: "https://aclanthology.org/2024.naacl-long.282/"
categories: ['efficient-transformer-optimization-techniques']
tags: ['lora', 'meta-learning', 'parameter-efficient-finetuning']
venue: "NAACL 2024"
tldr: "Proposes AutoLoRA, a method to automatically tune matrix ranks in Low-Rank Adaptation using meta-learning."
---

# AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning

**Source**: [https://aclanthology.org/2024.naacl-long.282/](https://aclanthology.org/2024.naacl-long.282/)

**TLDR**: Proposes AutoLoRA, a method to automatically tune matrix ranks in Low-Rank Adaptation using meta-learning.

## Abstract

AbstractLarge-scale pretraining followed by task-specific finetuning has achieved great success in various NLP tasks. Since finetuning all parameters of large pretrained models poses substantial computational and memory challenges, several efficient finetuning methods have been developed. Among them, low-rank adaptation (LoRA), which finetunes low-rank incremental update matrices on top of frozen pretrained weights, has proven particularly effective. Nonetheless, LoRA’s uniform rank assignment across all layers, along with its reliance on an exhaustive search to find the best rank, leads to high computation costs and suboptimal finetuning performance. To address these limitations, we introduce AutoLoRA, a meta learning based framework for automatically identifying the optimal rank of each LoRA layer. AutoLoRA associates each rank-1 matrix in a low-rank update matrix with a selection variable, which determines whether the rank-1 matrix should be discarded. A meta learning based method is developed to learn these selection variables. The optimal rank is determined by thresholding the values of these variables. Our comprehensive experiments on natural language understanding, generation, and sequence labeling demonstrate the effectiveness of AutoLoRA. The code is publicly available at https://github.com/ruz048/AutoLoRA