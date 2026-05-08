---
title: "PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning"
source: "https://aclanthology.org/2024.naacl-long.177/"
categories: ['efficient-transformer-optimization-techniques', 'llm-backdoor-attacks-defense']
tags: ['backdoor-attack', 'adversarial-tuning', 'prompting', 'security']
venue: "NAACL 2024"
tldr: "A few-shot method (PromptFix) removes backdoors in language models via adversarial prompt tuning."
---

# PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning

**Source**: [https://aclanthology.org/2024.naacl-long.177/](https://aclanthology.org/2024.naacl-long.177/)

**TLDR**: A few-shot method (PromptFix) removes backdoors in language models via adversarial prompt tuning.

## Abstract

AbstractPre-trained language models (PLMs) have attracted enormous attention over the past few years with their unparalleled performances. Meanwhile, the soaring cost to train PLMs as well as their amazing generalizability have jointly contributed to few-shot fine-tuning and prompting as the most popular training paradigms for natural language processing (NLP) models. Nevertheless, existing studies have shown that these NLP models can be backdoored such that model behavior is manipulated when trigger tokens are presented.In this paper, we propose PromptFix, a novel backdoor mitigation strategy for NLP models via adversarial prompt-tuning in few-shot settings.Unlike existing NLP backdoor removal methods, which rely on accurate trigger inversion and subsequent model fine-tuning, PromptFix keeps the model parameters intact and only utilizes two extra sets of soft tokens which approximate the trigger and counteract it respectively. The use of soft tokens and adversarial optimization eliminates the need to enumerate possible backdoor configurations and enables an adaptive balance between trigger finding and preservation of performance.Experiments with various backdoor attacks validate the effectiveness of the proposed method and the performances when domain shift is present further shows PromptFix’s applicability to models pretrained on unknown data source which is the common case in prompt tuning scenarios.