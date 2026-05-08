---
title: "Knowing What LLMs DO NOT Know: A Simple Yet Effective Self-Detection Method"
source: "https://aclanthology.org/2024.naacl-long.390/"
categories: ['large-language-model-evaluation-augmentation']
tags: ['self-detection', 'hallucination', 'uncertainty']
venue: "NAACL 2024"
tldr: "Introduces a self-detection method for LLMs to identify when they do not know an answer to reduce hallucinations."
---

# Knowing What LLMs DO NOT Know: A Simple Yet Effective Self-Detection Method

**Source**: [https://aclanthology.org/2024.naacl-long.390/](https://aclanthology.org/2024.naacl-long.390/)

**TLDR**: Introduces a self-detection method for LLMs to identify when they do not know an answer to reduce hallucinations.

## Abstract

AbstractLarge Language Models (LLMs) have shown great potential in Natural Language Processing (NLP) tasks.However, recent literature reveals that LLMs hallucinate intermittently, which impedes their reliability for further utilization. In this paper, we propose a novel self-detection method to detect which questions an LLM does not know.Our proposal is empirical and applicable for continually upgrading LLMs compared with state-of-the-art methods. Specifically, we examine the divergence of the LLM’s behaviors on different verbalizations for a question and examine the atypicality of the verbalized input. We combine the two components to identify whether the model generates a non-factual response to the question. The above components can be accomplished by utilizing the LLM itself without referring to any other external resources. We conduct comprehensive experiments and demonstrate the effectiveness of our method for recently released LLMs involving Llama 2, Vicuna, ChatGPT, and GPT-4 across factoid question-answering, arithmetic reasoning, and commonsense reasoning tasks.