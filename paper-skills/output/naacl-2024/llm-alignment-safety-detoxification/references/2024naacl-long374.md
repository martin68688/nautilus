---
title: "Intent-conditioned and Non-toxic Counterspeech Generation using Multi-Task Instruction Tuning with RLAIF"
source: "https://aclanthology.org/2024.naacl-long.374/"
categories: ['llm-backdoor-attacks-defense', 'llm-alignment-safety-detoxification']
tags: ['counterspeech', 'instruction-tuning', 'rl-aif']
venue: "NAACL 2024"
tldr: "Proposes a multi-task instruction tuning approach with RLAIF for generating intent-conditioned and non-toxic counterspeech."
---

# Intent-conditioned and Non-toxic Counterspeech Generation using Multi-Task Instruction Tuning with RLAIF

**Source**: [https://aclanthology.org/2024.naacl-long.374/](https://aclanthology.org/2024.naacl-long.374/)

**TLDR**: Proposes a multi-task instruction tuning approach with RLAIF for generating intent-conditioned and non-toxic counterspeech.

## Abstract

AbstractCounterspeech, defined as a response to mitigate online hate speech, is increasingly used as a non-censorial solution. The effectiveness of addressing hate speech involves dispelling the stereotypes, prejudices, and biases often subtly implied in brief, single-sentence statements or abuses. These expressions challenge language models, especially in seq2seq tasks, as model performance typically excels with longer contexts. Our study introduces CoARL, a novel framework enhancing counterspeech generation by modeling the pragmatic implications underlying social biases in hateful statements. The first two phases of CoARL involve sequential multi-instruction tuning, teaching the model to understand intents, reactions, and harms of offensive statements, and then learning task-specific low-rank adapter weights for generating intent-conditioned counterspeech. The final phase uses reinforcement learning to fine-tune outputs for effectiveness and nontoxicity. CoARL outperforms existing benchmarks in intent-conditioned counterspeech generation, showing an average improvement of ∼3 points in intent-conformity and ∼4 points in argument-quality metrics. Extensive human evaluation supports CoARL’s efficacy in generating superior and more context-appropriate responses compared to existing systems, including prominent LLMs like ChatGPT.