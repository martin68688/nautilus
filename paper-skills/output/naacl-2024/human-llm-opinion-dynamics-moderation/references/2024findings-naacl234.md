---
title: "WebWISE: Unlocking Web Interface Control for LLMs via Sequential Exploration"
source: "https://aclanthology.org/2024.findings-naacl.234/"
categories: ['human-llm-opinion-dynamics-moderation', 'strategic-reasoning-and-interactive-agents']
tags: ['web-interaction', 'sequential-exploration', 'llm-agents', 'software-tasks']
venue: "NAACL 2024"
tldr: "This paper investigates using large language models to automatically perform web software tasks via sequential exploration with click, scroll, and text input operations."
---

# WebWISE: Unlocking Web Interface Control for LLMs via Sequential Exploration

**Source**: [https://aclanthology.org/2024.findings-naacl.234/](https://aclanthology.org/2024.findings-naacl.234/)

**TLDR**: This paper investigates using large language models to automatically perform web software tasks via sequential exploration with click, scroll, and text input operations.

## Abstract

AbstractThis paper investigates using Large Language Models (LLMs) to automatically perform web software tasks using click, scroll, and text in- put operations. Previous approaches, such as reinforcement learning (RL) or imitation learning, are inefficient to train and task-specific. Our method uses filtered Document Object Model (DOM) elements as observations and performs tasks step-by-step, sequentially generating small programs based on the current observations. We use in-context learning, either benefiting from a single manually provided example, or an automatically generated example based on a successful zero-shot trial. We evaluate our proposed method on the MiniWob++ benchmark. With only one in-context example, our WebWISE method using gpt-3.5-turbo achieves similar or better performance than other methods that require many demonstrations or trials.