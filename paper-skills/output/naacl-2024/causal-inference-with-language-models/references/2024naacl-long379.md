---
title: "Instructing Large Language Models to Identify and Ignore Irrelevant Conditions"
source: "https://aclanthology.org/2024.naacl-long.379/"
categories: ['clinical-nlp-biomedical-applications', 'llm-evaluation-summarization-argument-extraction', 'causal-inference-with-language-models']
tags: ['math-word-problem', 'reasoning', 'irrelevant-information']
venue: "NAACL 2024"
tldr: "Instructs LLMs to identify and ignore irrelevant conditions when solving math word problems."
---

# Instructing Large Language Models to Identify and Ignore Irrelevant Conditions

**Source**: [https://aclanthology.org/2024.naacl-long.379/](https://aclanthology.org/2024.naacl-long.379/)

**TLDR**: Instructs LLMs to identify and ignore irrelevant conditions when solving math word problems.

## Abstract

AbstractMath word problem (MWP) solving requires generating a reasoning path based on a given problem description that often contains irrelevant conditions.Existing chain-of-thought (CoT) prompting methods elicited multi-step reasoning abilities of large language models (LLMs) to solve MWPs.However, they were seriously confused by the irrelevant conditions, resulting in low accuracy.In this paper, we propose a novel approach named I3C that instructs LLMs to identify and ignore irrelevant conditions.It identifies a set of irrelevant condition candidates that have a weak semantic relevance with the question.Then it prompts LLMs to verify the irrelevant conditions.Lastly it instructs the LLMs with the verification on relevant and irrelevant conditions to avoid confusion and improve reasoning paths.Moreover, we propose to select (problem, reasoning paths) pairs as demonstrations to enhance I3C with few-shot reasoning. We develop I3C-Select that selects the most confusing problems based on the semantic relevance measurement.We conduct extensive experiments on eight MWP datasets.I3C can be combined with any CoT prompting methods to improve the performance of solving MWPs.Notably, with GPT-3.5-Turbo and I3C-Select, we achieve an accuracy of 96.0 and 94.1 on GSM-IC2-1K and GSM-ICM-1K, respectively, significantly outperforming the state-of-the-art few-shot prompting method Complex-CoT by +11.7 and +11.1.Our implementation is made publicly available at https://wzy6642.github.io/I3C.github.io/.