---
title: "IterAlign: Iterative Constitutional Alignment of Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.78/"
categories: ['llm-alignment-safety-detoxification']
tags: ['alignment', 'constitutional-ai', 'iterative']
venue: "NAACL 2024"
tldr: "Introduces an iterative constitutional alignment method to improve LLM safety and reliability without human feedback."
---

# IterAlign: Iterative Constitutional Alignment of Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.78/](https://aclanthology.org/2024.naacl-long.78/)

**TLDR**: Introduces an iterative constitutional alignment method to improve LLM safety and reliability without human feedback.

## Abstract

AbstractWith the rapid development of large language models (LLMs), aligning LLMs with human values and societal norms to ensure their reliability and safety has become crucial. Reinforcement learning with human feedback (RLHF) and Constitutional AI (CAI) have been proposed for LLM alignment. However, these methods require either heavy human annotations or explicitly pre-defined constitutions, which are labor-intensive and resource-consuming. To overcome these drawbacks, we study constitution-based LLM alignment and propose a data-driven constitution discovery and self-alignment framework called IterAlign. IterAlign leverages red teaming to unveil the weaknesses of an LLM and automatically discovers new constitutions using a stronger LLM. These constitutions are then used to guide self-correction of the base LLM. Such a constitution discovery pipeline can be run iteratively and automatically to discover new constitutions that specifically target the alignment gaps in the current LLM. Empirical results on several safety benchmark datasets and multiple base LLMs show that IterAlign successfully improves truthfulness, helpfulness, harmlessness and honesty, improving the LLM alignment by up to 13.5% in harmlessness.