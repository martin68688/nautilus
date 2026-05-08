---
title: "BotChat: Evaluating LLMs’ Capabilities of Having Multi-Turn Dialogues"
source: "https://aclanthology.org/2024.findings-naacl.201/"
categories: ['large-language-model-evaluation-augmentation', 'llm-evaluation-summarization-opinion-analysis']
tags: ['dialogue_evaluation', 'multi-turn', 'LLM']
venue: "NAACL 2024"
tldr: "BotChat provides a formative assessment of current LLMs' capabilities in having high-quality, multi-turn dialogues."
---

# BotChat: Evaluating LLMs’ Capabilities of Having Multi-Turn Dialogues

**Source**: [https://aclanthology.org/2024.findings-naacl.201/](https://aclanthology.org/2024.findings-naacl.201/)

**TLDR**: BotChat provides a formative assessment of current LLMs' capabilities in having high-quality, multi-turn dialogues.

## Abstract

AbstractIn the realm of modern Large Language Models (LLMs), facilitating high-quality, multi-turn dialogues with humans represents a cornerstone feature. However, human-based evaluation of such a capability involves substantial manual effort. This study offers a formative assessment of current LLMs’ proficiency in emulating human-like, multi-turn conversations using an LLM-centric approach. The evaluation encompasses three key elements in the evaluation pipeline: utterance generation, evaluation protocol, and judgement, and we delve deeply into each aspect. GPT-4, both as an utterance generator and as a judge, exhibits exceptional performance. As a generator, GPT-4 crafts dialogues indistinguishable from human interactions in terms of style and flow. When judging, it shows a heightened alignment with human evaluative standards and consistency. Conversely, other LLMs face challenges in producing quality multi-turn dialogues, hindered by inadequate instruction-following abilities, a propensity for prolix utterances, and overall limited capabilities. Notably, generating extensive dialogues (e.g., spanning tens of turns) remains a formidable task for most LLMs, particularly in Chinese contexts. We hope that our work can serve as a valuable resource for evaluating the multi-turn chatting capabilities of LLMs. Related resources are available at https://github.com/open-compass/BotChat.