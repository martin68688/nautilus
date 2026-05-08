---
title: "MICo: Preventative Detoxification of Large Language Models through Inhibition Control"
source: "https://aclanthology.org/2024.findings-naacl.110/"
categories: ['llm-backdoor-attacks-defense', 'code-search-clone-detection']
tags: ['detoxification', 'preventative', 'alignment']
venue: "NAACL 2024"
tldr: "Proposes a preventative detoxification method for LLMs using inhibition control to prevent toxic degeneration during generation."
---

# MICo: Preventative Detoxification of Large Language Models through Inhibition Control

**Source**: [https://aclanthology.org/2024.findings-naacl.110/](https://aclanthology.org/2024.findings-naacl.110/)

**TLDR**: Proposes a preventative detoxification method for LLMs using inhibition control to prevent toxic degeneration during generation.

## Abstract

AbstractLarge Language Models (LLMs) are powerful tools which have been both dominant and commonplace in the field of Artificial Intelligence. Yet, LLMs have a tendency to devolve into toxic degeneration, wherein otherwise safe and unproblematic models begin generating toxic content. For the sake of social responsibility and inspired by the biological mechanisms of inhibition control, we introduce the paradigm of Education for Societal Norms (ESN). By collecting and labeling examples as acceptable and unacceptable (in this case toxic and non-toxic), and including a corresponding acceptable rewrite with every unacceptable example, we introduce a new mechanism for LLM detoxification. We annotate a dataset of 2,850 entries and use it to fine-tune a model, which we call a Model with Inhibition Control (MICo). Evaluating this model on toxicity detection capability, rewrite detoxification, meaning preservation, and overall toxicity reduction, we discover significant improvements over the baseline model. In our experiments we show that overall toxicity of this model is more than 60% reduced, with over 75% reduction in severe toxicity.