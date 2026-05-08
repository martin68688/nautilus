---
title: "SocREval: Large Language Models with the Socratic Method for Reference-free Reasoning Evaluation"
source: "https://aclanthology.org/2024.findings-naacl.175/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['llm-evaluation', 'reasoning', 'benchmark']
venue: "NAACL 2024"
tldr: "Introduces a reference-free evaluation method using LLMs and Socratic questioning to assess reasoning chains."
---

# SocREval: Large Language Models with the Socratic Method for Reference-free Reasoning Evaluation

**Source**: [https://aclanthology.org/2024.findings-naacl.175/](https://aclanthology.org/2024.findings-naacl.175/)

**TLDR**: Introduces a reference-free evaluation method using LLMs and Socratic questioning to assess reasoning chains.

## Abstract

AbstractTo comprehensively gauge the capacity of current models for complex reasoning, it is crucial to assess their step-by-step reasoning in a scalable manner. Established reference-based evaluation metrics rely on human-annotated reasoning chains as references to assess the model-derived chains. However, such “gold-standard” human-written reasoning chains may not be unique and their acquisition is often labor-intensive. Existing reference-free reasoning evaluation metrics, while eliminating the need for human-crafted reasoning chains as references, often require fine-tuning with human-derived chains before evaluation, complicating the process and questioning their adaptability to other datasets. To address these challenges, we harness GPT-4 to automatically evaluate reasoning chain quality, thereby removing the dependency on human-written reasoning chains for both model fine-tuning and evaluative purposes. Leveraging the Socratic method, we develop SocREval (**Soc**ratic Method-Inspired **R**easoning **Eval**uation), a novel approach for prompt design in reference-free reasoning evaluation. Empirical results from four human annotated datasets reveal that SocREval significantly improves GPT-4’s performance, surpassing existing reference-free and reference-based reasoning evaluation metrics. Beyond its demonstrated efficacy, SocREval, proves to be both cost-efficient and robust to prompt writing and example selection, as substantiated by our in-depth analysis.