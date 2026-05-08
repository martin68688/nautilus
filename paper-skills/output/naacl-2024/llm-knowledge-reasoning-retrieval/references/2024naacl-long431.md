---
title: "Naive Bayes-based Context Extension for Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.431/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['in-context-learning', 'context-extension', 'naive-bayes', 'llm']
venue: "NAACL 2024"
tldr: "Extends LLM context for in-context learning using a Naive Bayes approach to integrate more supervision examples."
---

# Naive Bayes-based Context Extension for Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.431/](https://aclanthology.org/2024.naacl-long.431/)

**TLDR**: Extends LLM context for in-context learning using a Naive Bayes approach to integrate more supervision examples.

## Abstract

AbstractLarge Language Models (LLMs) have shown promising in-context learning abilities. However, conventional In-Context Learning (ICL) approaches are often impeded by length limitations of transformer architecture, which pose challenges when attempting to effectively integrate supervision from a substantial number of demonstration examples. In this paper, we introduce a novel framework, called Naive Bayes-based Context Extension (NBCE), to enable existing LLMs to perform ICL with an increased number of demonstrations by significantly expanding their context size. Importantly, this expansion does not require fine-tuning or dependence on particular model architectures, all the while preserving linear efficiency. NBCE initially splits the context into equal-sized windows fitting the target LLM’s maximum length. Then, it introduces a voting mechanism to select the most relevant window, regarded as the posterior context. Finally, it employs Bayes’ theorem to generate the test task. Our experimental results demonstrate that NBCE substantially enhances performance, particularly as the number of demonstration examples increases, consistently outperforming alternative methods. The NBCE code will be made publicly accessible. The code NBCE is available at: https://github.com/amurtadha/NBCE-master