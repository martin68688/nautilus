---
title: "Parameter-Efficient Instruction Tuning of Large Language Models For Extreme Financial Numeral Labelling"
source: "https://aclanthology.org/2024.naacl-long.410/"
categories: ['llm-knowledge-reasoning-retrieval', 'clinical-nlp-biomedical-applications']
tags: ['financial-nlp', 'instruction-tuning', 'extreme-classification']
venue: "NAACL 2024"
tldr: "Investigates parameter-efficient instruction tuning of LLMs for extreme classification of financial numerals using a generative paradigm."
---

# Parameter-Efficient Instruction Tuning of Large Language Models For Extreme Financial Numeral Labelling

**Source**: [https://aclanthology.org/2024.naacl-long.410/](https://aclanthology.org/2024.naacl-long.410/)

**TLDR**: Investigates parameter-efficient instruction tuning of LLMs for extreme classification of financial numerals using a generative paradigm.

## Abstract

AbstractWe study the problem of automatically annotating relevant numerals (GAAP metrics) occurring in the financial documents with their corresponding XBRL tags. Different from prior works, we investigate the feasibility of solving this extreme classification problem using a generative paradigm through instruction tuning of Large Language Models (LLMs). To this end, we leverage metric metadata informationto frame our target outputs while proposing a parameter efficient solution for the task using LoRA. We perform experiments on two recently released financial numeric labeling datasets. Our proposed model, **FLAN-FinXC**, achieves new state-of-the-art performances on both the datasets, outperforming several strong baselines. We explain the better scores of our proposed model by demonstrating its capability for zero-shot as well as the least frequently occurring tags. Also, even when we fail to predict the XBRL tags correctly, our generated output has substantial overlap with the ground-truth in majority of the cases.