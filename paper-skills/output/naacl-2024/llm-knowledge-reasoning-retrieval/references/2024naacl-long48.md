---
title: "Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model"
source: "https://aclanthology.org/2024.naacl-long.48/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-alignment-safety-detoxification', 'code-search-clone-detection']
tags: ['tool-use', 'chain-of-solving', 'open-source-llm']
venue: "NAACL 2024"
tldr: "Toolink links toolkit creation and use through a chain-of-solving approach on open-source language models."
---

# Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model

**Source**: [https://aclanthology.org/2024.naacl-long.48/](https://aclanthology.org/2024.naacl-long.48/)

**TLDR**: Toolink links toolkit creation and use through a chain-of-solving approach on open-source language models.

## Abstract

AbstractLarge Language Models (LLMs) have demonstrated remarkable progress in utilizing tools, but their closed-source nature and high inference costs pose limitations on their adaptability, necessitating a valid method that leverages smaller, open-sourced models. In this paper, we introduce Toolink, a comprehensive framework that performs task-solving by first creating a toolkit and then integrating the planning and calling of tools through a chain-of-solving (CoS) approach. We first validate the efficacy of Toolink in harnessing the model’s creativity and CoS ability on ChatGPT. Subsequently, we curate CoS-GPT, a chain-of-solving dataset designed for tool-using, and finetune the LLaMA-7B model. It results in LLaMA-CoS, a powerful open-source model with advanced tool-planning and tool-calling capabilities. Evaluation of diverse tasks from BIG-bench demonstrates its CoS ability matches that of ChatGPT while its performance surpasses the chain-of-thought approach. Further studies highlight the generalization of LLaMA-CoS to unseen tasks and showcase its capability in using toolkits not explicitly tailored for the target task, affirming its robustness in real-world scenarios. All codes and data are released.