---
title: "Reverse Chain: A Generic-Rule for LLMs to Master Multi-API Planning"
source: "https://aclanthology.org/2024.findings-naacl.22/"
categories: ['bpe-subword-parsing-evaluation', 'llm-ranking-benchmarking-adaptation']
tags: ['api-planning', 'function-calling', 'reasoning']
venue: "NAACL 2024"
tldr: "Introduces a generic rule to improve LLMs' ability to plan and execute multi-API function calls in a context-learning setting."
---

# Reverse Chain: A Generic-Rule for LLMs to Master Multi-API Planning

**Source**: [https://aclanthology.org/2024.findings-naacl.22/](https://aclanthology.org/2024.findings-naacl.22/)

**TLDR**: Introduces a generic rule to improve LLMs' ability to plan and execute multi-API function calls in a context-learning setting.

## Abstract

AbstractWhile enabling large language models to implement function calling (known as APIs) can greatly enhance the performance of Large Language Models (LLMs), function calling is still a challenging task due to the complicated relations between different APIs, especially in a context-learning setting without fine-tuning. This paper introduces “Reverse Chain”, a controllable, target-driven approach designed to empower LLMs with the capability to operate external APIs only via prompts. Recognizing that most LLMs have limited tool-use capabilities, Reverse Chain limits LLMs to executing simple tasks, e.g., API Selection and Argument Completion. Furthermore, to manage a controllable multi-function calling, Reverse Chain adopts a generic rule-based on a backward reasoning process. This rule determines when to do API selection or Argument completion. To evaluate the multi-tool-use capability of LLMs, we have released a compositional multi-tool task dataset, available at https://github.com/zhangyingerjelly/reverse-chain. Extensive numerical experiments validate the remarkable proficiency of Reverse Chain in managing multiple API calls.