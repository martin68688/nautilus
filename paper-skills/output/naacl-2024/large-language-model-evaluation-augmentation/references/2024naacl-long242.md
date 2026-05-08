---
title: "Rectifying Demonstration Shortcut in In-Context Learning"
source: "https://aclanthology.org/2024.naacl-long.242/"
categories: ['large-language-model-evaluation-augmentation', 'causal-inference-with-language-models']
tags: ['in-context-learning', 'shortcut', 'reasoning', 'llm']
venue: "NAACL 2024"
tldr: "A method to rectify LLMs' reliance on demonstration shortcuts in ICL by encouraging learning of input-label relationships."
---

# Rectifying Demonstration Shortcut in In-Context Learning

**Source**: [https://aclanthology.org/2024.naacl-long.242/](https://aclanthology.org/2024.naacl-long.242/)

**TLDR**: A method to rectify LLMs' reliance on demonstration shortcuts in ICL by encouraging learning of input-label relationships.

## Abstract

AbstractLarge language models (LLMs) are able to solve various tasks with only a few demonstrations utilizing their in-context learning (ICL) abilities.However, LLMs often rely on their pre-trained semantic priors of demonstrations rather than on the input-label relationships to proceed with ICL prediction. In this work, we term this phenomenon as the ‘Demonstration Shortcut’.While previous works have primarily focused on improving ICL prediction results for predefined tasks, we aim to rectify the Demonstration Shortcut, thereby enabling the LLM to effectively learn new input-label relationships from demonstrations.To achieve this, we introduce In-Context Calibration, a demonstration-aware calibration method.We evaluate the effectiveness of the proposed method in two settings: (1) the Original ICL Task using the standard label space and (2) the Task Learning setting, where the label space is replaced with semantically unrelated tokens.In both settings, In-Context Calibration demonstrates substantial improvements, with results generalized across three LLM families (OPT, GPT, and Llama2) under various configurations.