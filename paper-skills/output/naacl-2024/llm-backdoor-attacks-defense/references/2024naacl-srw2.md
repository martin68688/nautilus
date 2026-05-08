---
title: "Rephrasing Invokes Better Generations for Large Language Models"
source: "https://aclanthology.org/2024.naacl-srw.2/"
categories: ['large-language-model-evaluation-augmentation', 'llm-backdoor-attacks-defense']
tags: ['prompt-engineering', 'generation', 'rephrasing']
venue: "NAACL 2024"
tldr: "Shows that rephrasing user queries before feeding them to an LLM can improve the quality of the generated outputs."
---

# Rephrasing Invokes Better Generations for Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-srw.2/](https://aclanthology.org/2024.naacl-srw.2/)

**TLDR**: Shows that rephrasing user queries before feeding them to an LLM can improve the quality of the generated outputs.

## Abstract

AbstractIn the realm of emerging multitasking abilities of Large language models (LLMs), methodologies like prompt tuning enable low-cost adaptation to downstream tasks without retraining the model. However, automatic input pre-processing when LLMs are unavailable is currently under-studied. This paper proposes ReLLM (Rephrasing for LLMs), a method that automatically paraphrases input content for better output generations. ReLLM replaces low-frequency lexical items with their high-frequency counterparts. This substitution is particularly beneficial for low-resource language tasks that lack sufficient training data and resources. ReLLM is user-friendly and requires no additional LLM training. Experimental results in cross-lingual summarization, and natural language inference demonstrate the effectiveness of ReLLM.