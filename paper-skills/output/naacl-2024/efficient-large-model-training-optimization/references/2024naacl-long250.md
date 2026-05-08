---
title: "Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Model"
source: "https://aclanthology.org/2024.naacl-long.250/"
categories: ['efficient-large-model-training-optimization', 'legal-ai-nlp-applications']
tags: ['distillation', 'paraphrasing', 'summarization', 'data-generation']
venue: "NAACL 2024"
tldr: "Proposes a distillation framework to create high-quality paraphrasing/summarization data from a low-quality model."
---

# Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Model

**Source**: [https://aclanthology.org/2024.naacl-long.250/](https://aclanthology.org/2024.naacl-long.250/)

**TLDR**: Proposes a distillation framework to create high-quality paraphrasing/summarization data from a low-quality model.

## Abstract

AbstractWe present Impossible Distillation, a novel framework for paraphrasing and sentence summarization, that distills a high-quality dataset and model from a low-quality teacher that itself cannot perform these tasks. Unlike prior works that rely on an extreme-scale teacher model (e.g., GPT3) or task-specific architecture, we hypothesize and verify the paraphrastic proximity intrinsic to pre-trained LMs (e.g., GPT2), where paraphrases occupy a proximal subspace in the LM distribution. By identifying and distilling generations from these subspaces, Impossible Distillation produces a high-quality dataset and model even from GPT2-scale LMs. We evaluate our method on multiple benchmarks spanning unconstrained / syntax-controlled paraphrase generation and sentence summarization. Our model with 770M parameters consistently outperforms strong baselines, including models distilled from ChatGPT, and sometimes, even ChatGPT itself. Also, we find that our distilled dataset from 1.5B LMs exhibits higher diversity and fidelity than up to 13 times larger datasets.