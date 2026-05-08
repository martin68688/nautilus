---
title: "LETI: Learning to Generate from Textual Interactions"
source: "https://aclanthology.org/2024.findings-naacl.16/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['fine-tuning', 'textual-interaction', 'language-model']
venue: "NAACL 2024"
tldr: "Explores a method for fine-tuning language models by learning from textual interactions, such as feedback or corrections, rather than just input-output pairs."
---

# LETI: Learning to Generate from Textual Interactions

**Source**: [https://aclanthology.org/2024.findings-naacl.16/](https://aclanthology.org/2024.findings-naacl.16/)

**TLDR**: Explores a method for fine-tuning language models by learning from textual interactions, such as feedback or corrections, rather than just input-output pairs.

## Abstract

AbstractFine-tuning pre-trained language models (LMs) is essential for enhancing their capabilities.Existing techniques commonly fine-tune on input-output pairs (e.g., instruction tuning) or with numerical rewards that gauge the output quality (e.g., RLHF). We explore LMs’ potential to **le**arn from **t**extual **i**nteractions (**LETI**) that not only check their correctness with *binary labels* but also pinpoint and explain errors in their outputs through *textual feedback*.Our focus is the code generation task, where the model produces code based on natural language instructions. This setting invites a natural and scalable way to acquire textual feedback: the error messages and stack traces from code execution using a Python interpreter. LETI iteratively fine-tunes the model, using the LM objective, on a concatenation of natural language instructions, LM-generated programs, and textual feedback. Prepended to this fine-tuning text, a binary reward token is used to differentiate correct and buggy solutions.LETI requires *no* ground-truth outputs for training and even outperforms a fine-tuned baseline that does. LETI not only improves the performance of LMs on a code generation dataset MBPP, but also generalizes to other datasets. Trained on MBPP, it achieves comparable or better performance than the base LMs on unseen problems in HumanEval. Furthermore, compared to binary feedback, we observe that textual feedback leads to improved generation quality and sample efficiency, achieving the same performance with fewer than half of the gradient steps.LETI is equally applicable in natural language tasks when they can be formulated as code generation, which we empirically verified on event argument extraction.