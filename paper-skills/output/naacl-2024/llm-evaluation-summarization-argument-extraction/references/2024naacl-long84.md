---
title: "A Symbolic Framework for Evaluating Mathematical Reasoning and Generalisation with Transformers"
source: "https://aclanthology.org/2024.naacl-long.84/"
categories: ['llm-evaluation-summarization-argument-extraction', 'latent-space-mathematical-derivations']
tags: ['mathematical-reasoning', 'symbolic', 'generalization', 'transformers']
venue: "NAACL 2024"
tldr: "Proposes a symbolic framework to generate and perturb equation derivations at scale to evaluate Transformer generalization on out-of-distribution math problems."
---

# A Symbolic Framework for Evaluating Mathematical Reasoning and Generalisation with Transformers

**Source**: [https://aclanthology.org/2024.naacl-long.84/](https://aclanthology.org/2024.naacl-long.84/)

**TLDR**: Proposes a symbolic framework to generate and perturb equation derivations at scale to evaluate Transformer generalization on out-of-distribution math problems.

## Abstract

AbstractThis paper proposes a methodology for generating and perturbing detailed derivations of equations at scale, aided by a symbolic engine, to evaluate the generalisability of Transformers to out-of-distribution mathematical reasoning problems. Instantiating the framework in the context of sequence classification tasks, we compare the capabilities of GPT-4, GPT-3.5, and a canon of fine-tuned BERT models, exploring the relationship between specific operators and generalisation failure via the perturbation of reasoning aspects such as symmetry and variable surface forms. Surprisingly, our empirical evaluation reveals that the average in-distribution performance of fine-tuned models surpasses GPT-3.5, and rivals GPT-4. However, perturbations to input reasoning can reduce their performance by up to 80 F1 points. Overall, the results suggest that the in-distribution performance of smaller open-source models may potentially rival GPT by incorporating appropriately structured derivation dependencies during training, and highlight a shared weakness between BERT and GPT involving a relative inability to decode indirect references to mathematical entities. We release the full codebase, constructed datasets, and fine-tuned models to encourage future progress in the field.