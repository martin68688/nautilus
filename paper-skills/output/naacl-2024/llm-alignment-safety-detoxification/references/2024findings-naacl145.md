---
title: "Role Prompting Guided Domain Adaptation with General Capability Preserve for Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.145/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-alignment-safety-detoxification']
tags: ['domain-adaptation', 'catastrophic-forgetting', 'prompting']
venue: "NAACL 2024"
tldr: "Uses role prompting to adapt LLMs to specialized domains while preserving their general capabilities and preventing catastrophic forgetting."
---

# Role Prompting Guided Domain Adaptation with General Capability Preserve for Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.145/](https://aclanthology.org/2024.findings-naacl.145/)

**TLDR**: Uses role prompting to adapt LLMs to specialized domains while preserving their general capabilities and preventing catastrophic forgetting.

## Abstract

AbstractThe growing interest in Large Language Models (LLMs) for specialized applications has revealed a significant challenge: when tailored to specific domains, LLMs tend to experience catastrophic forgetting, compromising their general capabilities and leading to a suboptimal user experience. Additionally, crafting a versatile model for multiple domains simultaneously often results in a decline in overall performance due to confusion between domains. In response to these issues, we present the RolE Prompting Guided Multi-Domain Adaptation (REGA) strategy. This novel approach effectively manages multi-domain LLM adaptation through three key components: 1) Self-Distillation constructs and replays general-domain exemplars to alleviate catastrophic forgetting. 2) Role Prompting assigns a central prompt to the general domain and a unique role prompt to each specific domain to minimize inter-domain confusion during training. 3) Role Integration reuses and integrates a small portion of domain-specific data to the general-domain data, which are trained under the guidance of the central prompt. The central prompt is used for a streamlined inference process, removing the necessity to switch prompts for different domains.Empirical results demonstrate that REGA effectively alleviates catastrophic forgetting and inter-domain confusion. This leads to improved domain-specific performance compared to standard fine-tuned models, while still preserving robust general capabilities.