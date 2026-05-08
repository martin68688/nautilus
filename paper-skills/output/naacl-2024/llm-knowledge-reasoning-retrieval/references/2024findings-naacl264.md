---
title: "ADaPT: As-Needed Decomposition and Planning with Language Models"
source: "https://aclanthology.org/2024.findings-naacl.264/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['planning', 'decomposition', 'agent']
venue: "NAACL 2024"
tldr: "Proposes an as-needed decomposition and planning method for LLM-based interactive agents."
---

# ADaPT: As-Needed Decomposition and Planning with Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.264/](https://aclanthology.org/2024.findings-naacl.264/)

**TLDR**: Proposes an as-needed decomposition and planning method for LLM-based interactive agents.

## Abstract

AbstractLarge Language Models (LLMs) are increasingly being used for interactive decision-making tasks requiring planning and adapting to the environment. Recent works employ LLMs-as-agents in broadly two ways: iteratively determining the next action (iterative executors) or generating plans and executing sub-tasks using LLMs (plan-and-execute). However, these methods struggle with task complexity, as the inability to execute any sub-task may lead to task failure. To address these shortcomings, we introduce As-Needed Decomposition and Planning for complex Tasks (ADaPT), an approach that explicitly plans and decomposes complex sub-tasks as-needed, i.e., when the LLM is unable to execute them. ADaPT recursively decomposes sub-tasks to adapt to both task complexity and LLM capability. Our results demonstrate that ADaPT substantially outperforms established strong baselines, achieving success rates up to 28.3% higher in ALFWorld, 27% in WebShop, and 33% in TextCraft – a novel compositional dataset that we introduce. Through extensive analysis, we illustrate the importance of multilevel decomposition and establish that ADaPT dynamically adjusts to the capabilities of the executor LLM as well as to task complexity.