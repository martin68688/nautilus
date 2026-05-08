---
title: "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning"
source: "https://aclanthology.org/2024.naacl-long.421/"
categories: ['efficient-large-model-training-optimization', 'large-language-model-evaluation-augmentation']
tags: ['instruction-tuning', 'data-selection', 'self-guided', 'llm']
venue: "NAACL 2024"
tldr: "Introduces a self-guided method for LLMs to autonomously select high-quality instruction data from open-source datasets to boost performance."
---

# From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning

**Source**: [https://aclanthology.org/2024.naacl-long.421/](https://aclanthology.org/2024.naacl-long.421/)

**TLDR**: Introduces a self-guided method for LLMs to autonomously select high-quality instruction data from open-source datasets to boost performance.

## Abstract

AbstractIn the realm of Large Language Models (LLMs), the balance between instruction data quality and quantity is a focal point. Recognizing this, we introduce a self-guided methodology for LLMs to autonomously discern and select cherry samples from open-source datasets, effectively minimizing manual curation and potential cost for instruction tuning an LLM. Our key innovation, the Instruction-Following Difficulty (IFD) metric, emerges as a pivotal metric to identify discrepancies between a model’s expected responses and its intrinsic generation capability. Through the application of IFD, cherry samples can be pinpointed, leading to a marked uptick in model training efficiency. Empirical validations on datasets like Alpaca and WizardLM underpin our findings; with a mere 10% of original data input, our strategy showcases improved results. This synthesis of self-guided cherry-picking and the IFD metric signifies a transformative leap in the instruction tuning of LLMs, promising both efficiency and resource-conscious advancements. Codes, data, and models are available.