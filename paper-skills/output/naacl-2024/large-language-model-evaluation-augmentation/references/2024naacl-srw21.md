---
title: "Distilling Text Style Transfer With Self-Explanation From LLMs"
source: "https://aclanthology.org/2024.naacl-srw.21/"
categories: ['contrastive-and-generative-representation-learning', 'large-language-model-evaluation-augmentation']
tags: ['style-transfer', 'distillation', 'chain-of-thought', 'explanation']
venue: "NAACL 2024"
tldr: "Proposes a framework using LLMs and chain-of-thought prompting to distill text style transfer models with self-explanations."
---

# Distilling Text Style Transfer With Self-Explanation From LLMs

**Source**: [https://aclanthology.org/2024.naacl-srw.21/](https://aclanthology.org/2024.naacl-srw.21/)

**TLDR**: Proposes a framework using LLMs and chain-of-thought prompting to distill text style transfer models with self-explanations.

## Abstract

AbstractText Style Transfer (TST) seeks to alter the style of text while retaining its core content. Given the constraints of limited parallel datasets for TST, we propose CoTeX, a framework that leverages large language models (LLMs) alongside chain-of-thought (CoT) prompting to facilitate TST. CoTeX distills the complex rewriting and reasoning capabilities of LLMs into more streamlined models capable of working with both non-parallel and parallel data. Through experimentation across four TST datasets, CoTeX is shown to surpass traditional supervised fine-tuning and knowledge distillation methods, particularly in low-resource settings. We conduct a comprehensive evaluation, comparing CoTeX against current unsupervised, supervised, in-context learning (ICL) techniques, and instruction-tuned LLMs. Furthermore, CoTeX distinguishes itself by offering transparent explanations for its style transfer process.