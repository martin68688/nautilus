---
title: "LLM-based Frameworks for API Argument Filling in Task-Oriented Conversational Systems"
source: "https://aclanthology.org/2024.naacl-industry.36/"
categories: ['bpe-subword-parsing-evaluation', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['task-oriented-dialogue', 'api-argument-filling', 'llm-frameworks']
venue: "NAACL 2024"
tldr: "Develops LLM-based frameworks to improve the argument filling step for external APIs in task-oriented conversational systems."
---

# LLM-based Frameworks for API Argument Filling in Task-Oriented Conversational Systems

**Source**: [https://aclanthology.org/2024.naacl-industry.36/](https://aclanthology.org/2024.naacl-industry.36/)

**TLDR**: Develops LLM-based frameworks to improve the argument filling step for external APIs in task-oriented conversational systems.

## Abstract

AbstractTask-orientated conversational agents interact with users and assist them via leveraging external APIs. A typical task-oriented conversational system can be broken down into three phases: external API selection, argument filling, and response generation. The focus of our work is the task of argument filling, which is in charge of accurately providing arguments required by the selected API. Upon comprehending the dialogue history and the pre-defined API schema, the argument filling task is expected to provide the external API with the necessary information to generate a desirable agent action. In this paper, we study the application of Large Language Models (LLMs) for the problem of API argument filling task. Our initial investigation reveals that LLMs require an additional grounding process to successfully perform argument filling, inspiring us to design training and prompting frameworks to ground their responses. Our experimental results demonstrate that when paired with proposed techniques, the argument filling performance of LLMs noticeably improves, paving a new way toward building an automated argument filling framework.