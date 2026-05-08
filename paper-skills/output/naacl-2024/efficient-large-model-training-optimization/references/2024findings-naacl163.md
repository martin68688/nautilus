---
title: "Towards an On-device Agent for Text Rewriting"
source: "https://aclanthology.org/2024.findings-naacl.163/"
categories: ['efficient-large-model-training-optimization', 'llm-edge-distillation']
tags: ['on-device', 'text-rewriting', 'distillation']
venue: "NAACL 2024"
tldr: "Presents solutions for creating a smaller, potent on-device language model for text rewriting, addressing data cost and emergent capabilities."
---

# Towards an On-device Agent for Text Rewriting

**Source**: [https://aclanthology.org/2024.findings-naacl.163/](https://aclanthology.org/2024.findings-naacl.163/)

**TLDR**: Presents solutions for creating a smaller, potent on-device language model for text rewriting, addressing data cost and emergent capabilities.

## Abstract

AbstractLarge Language Models (LLMs) have demonstrated impressive capabilities for text rewriting. However creating a smaller yet potent language model for text rewriting presents two formidable challenges: costly data collection and absence of emergent capabilities.In this paper we present solutions to address the above challenges.We propose an new instruction tuning method to develop a mo-bile text rewriting model that leverages LLM-generated data and heuristic reinforcement learning, eliminating the need for human data collection. Moreover, to bridge the performance gap from the constraint size, we pro-pose a cascading approach based on the confidence levels which are distilled from the large server model’s critiques. To evaluate the text rewriting tasks for mobile scenarios, we introduce MessageRewriteEval, a human-labeled benchmark that focuses on text rewriting of messages through natural language instructions. Through empirical experiments, we demonstrate that our on-device model surpasses the current state-of-the-art LLMs in text rewriting while maintaining a significantly reduced model size using public benchmark EditEval and our new benchmark. We also demonstrate that our proposed cascading approach improves model performance further.