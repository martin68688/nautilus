---
title: "Language Models are Alignable Decision-Makers: Dataset and Application to the Medical Triage Domain"
source: "https://aclanthology.org/2024.naacl-industry.18/"
categories: ['efficient-llm-training-optimization', 'efficient-transformer-acceleration-techniques']
tags: ['data-curation', 'pretraining', 'model-evaluation', 'hyperparameter-tuning']
venue: "NAACL 2024"
tldr: "Studies the impact of pretraining data attributes (age, toxicity, domain) on model performance and introduces tuning curves with confidence bands for robust hyperparameter comparison."
---

# Language Models are Alignable Decision-Makers: Dataset and Application to the Medical Triage Domain

**Source**: [https://aclanthology.org/2024.naacl-industry.18/](https://aclanthology.org/2024.naacl-industry.18/)

**TLDR**: Studies the impact of pretraining data attributes (age, toxicity, domain) on model performance and introduces tuning curves with confidence bands for robust hyperparameter comparison.

## Abstract

AbstractIn difficult decision-making scenarios, it is common to have conflicting opinions among expert human decision-makers as there may not be a single right answer. Such decisions may be guided by different attributes that can be used to characterize an individual’s decision. We introduce a novel dataset for medical triage decision-making, labeled with a set of decision-maker attributes (DMAs). This dataset consists of 62 scenarios, covering six different DMAs, including ethical principles such as fairness and moral desert. We present a novel software framework for human-aligned decision-making by utilizing these DMAs, paving the way for trustworthy AI with better guardrails. Specifically, we demonstrate how large language models (LLMs) can serve as ethical decision-makers, and how their decisions can be aligned to different DMAs using zero-shot prompting. Our experiments focus on different open-source models with varying sizes and training techniques, such as Falcon, Mistral, and Llama 2. Finally, we also introduce a new form of weighted self-consistency that improves the overall quantified performance. Our results provide new research directions in the use of LLMs as alignable decision-makers. The dataset and open-source software are publicly available at: https://github.com/ITM-Kitware/llm-alignable-dm.