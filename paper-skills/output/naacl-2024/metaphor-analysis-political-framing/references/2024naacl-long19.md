---
title: "kNN-ICL: Compositional Task-Oriented Parsing Generalization with Nearest Neighbor In-Context Learning"
source: "https://aclanthology.org/2024.naacl-long.19/"
categories: ['metaphor-analysis-political-framing', 'llm-evaluation-summarization-argument-extraction', 'llm-ranking-benchmarking-adaptation']
tags: ['in-context-learning', 'task-oriented-parsing', 'knn', 'compositional-generalization']
venue: "NAACL 2024"
tldr: "Uses k-nearest neighbor in-context learning for compositional generalization in task-oriented parsing."
---

# kNN-ICL: Compositional Task-Oriented Parsing Generalization with Nearest Neighbor In-Context Learning

**Source**: [https://aclanthology.org/2024.naacl-long.19/](https://aclanthology.org/2024.naacl-long.19/)

**TLDR**: Uses k-nearest neighbor in-context learning for compositional generalization in task-oriented parsing.

## Abstract

AbstractTask-Oriented Parsing (TOP) enables conversational assistants to interpret user commands expressed in natural language, transforming them into structured outputs that combine elements of both natural language and intent/slot tags. Recently, Large Language Models (LLMs) have achieved impressive performance in synthesizing computer programs based on a natural-language prompt, mitigating the gap between natural language and structured programs. Our paper focuses on harnessing the capabilities of LLMs for semantic parsing tasks, addressing the following three key research questions: 1) How can LLMs be effectively utilized for semantic parsing tasks? 2) What defines an effective prompt? and 3) How can LLM overcome the length constraint and streamline prompt design by including all examples as prompts? We introduce k Nearest Neighbor In-Context Learning (kNN-ICL), which simplifies prompt engineering by allowing it to be built on top of any design strategy while providing access to all demo examples. Extensive experiments show that: 1) Simple ICL without kNN search can achieve a comparable performance with strong supervised models on the TOP tasks, and 2) kNN-ICL significantly improves the comprehension of complex requests by seamlessly integrating ICL with a nearest-neighbor approach. Notably, this enhancement is achieved without the need for additional data or specialized prompts.