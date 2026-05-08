---
title: "Conformal Intent Classification and Clarification for Fast and Accurate Intent Recognition"
source: "https://aclanthology.org/2024.findings-naacl.156/"
categories: ['contrastive-and-generative-representation-learning', 'legal-ai-nlp-applications']
tags: ['dialogue', 'intent-classification', 'conformal-prediction', 'clarification']
venue: "NAACL 2024"
tldr: "A conformal framework for fast, accurate intent classification that uses guaranteed clarification questions for uncertain inputs."
---

# Conformal Intent Classification and Clarification for Fast and Accurate Intent Recognition

**Source**: [https://aclanthology.org/2024.findings-naacl.156/](https://aclanthology.org/2024.findings-naacl.156/)

**TLDR**: A conformal framework for fast, accurate intent classification that uses guaranteed clarification questions for uncertain inputs.

## Abstract

AbstractWe present Conformal Intent Classification and Clarification (CICC), a framework for fast and accurate intent classification for task-oriented dialogue systems. The framework turns heuristic uncertainty scores of any intent classifier into a clarification question that is guaranteed to contain the true intent at a pre-defined confidence level.By disambiguating between a small number of likely intents, the user query can be resolved quickly and accurately. Additionally, we propose to augment the framework for out-of-scope detection.In a comparative evaluation using seven intent recognition datasets we find that CICC generates small clarification questions and is capable of out-of-scope detection.CICC can help practitioners and researchers substantially in improving the user experience of dialogue agents with specific clarification questions.