---
title: "Knowledgeable In-Context Tuning: Exploring and Exploiting Factual Knowledge for In-Context Learning"
source: "https://aclanthology.org/2024.findings-naacl.207/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction', 'causal-inference-with-language-models']
tags: ['in-context-learning', 'knowledge', 'reasoning']
venue: "NAACL 2024"
tldr: "Demonstrates that factual knowledge is crucial for in-context learning and proposes a method to explore and exploit it for improved ICL."
---

# Knowledgeable In-Context Tuning: Exploring and Exploiting Factual Knowledge for In-Context Learning

**Source**: [https://aclanthology.org/2024.findings-naacl.207/](https://aclanthology.org/2024.findings-naacl.207/)

**TLDR**: Demonstrates that factual knowledge is crucial for in-context learning and proposes a method to explore and exploit it for improved ICL.

## Abstract

AbstractLarge language models (LLMs) enable in-context learning (ICL) by conditioning on a few labeled training examples as a text-based prompt, eliminating the need for parameter updates and achieving competitive performance. In this paper, we demonstrate that factual knowledge is imperative for the performance of ICL in three core facets: the inherent knowledge learned in LLMs, the factual knowledge derived from the selected in-context examples, and the knowledge biases in LLMs for output generation. To unleash the power of LLMs in few-shot learning scenarios, we introduce a novel Knowledgeable In-Context Tuning (KICT) framework to further improve the performance of ICL:1) injecting knowledge into LLMs during continual self-supervised pre-training, 2) judiciously selecting the examples for ICL with high knowledge relevance, and 3) calibrating the prediction results based on prior knowledge.We evaluate the proposed approaches on autoregressive models (e.g., GPT-style LLMs) over multiple text classification and question-answering tasks. Experimental results demonstrate that KICT substantially outperforms strong baselines and improves by more than 13% and 7% on text classification and question-answering tasks, respectively.