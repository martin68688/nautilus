---
title: "Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation"
source: "https://aclanthology.org/2024.naacl-long.83/"
categories: ['language-model-cultural-linguistic-evaluation', 'empathetic-conversational-ai-counseling']
tags: ['evaluation', 'conversational-recommendation', 'user-simulation']
venue: "NAACL 2024"
tldr: "Introduces a protocol to evaluate LLMs as generative user simulators for conversational recommendation, assessing their ability to represent diverse users."
---

# Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation

**Source**: [https://aclanthology.org/2024.naacl-long.83/](https://aclanthology.org/2024.naacl-long.83/)

**TLDR**: Introduces a protocol to evaluate LLMs as generative user simulators for conversational recommendation, assessing their ability to represent diverse users.

## Abstract

AbstractSynthetic users are cost-effective proxies for real users in the evaluation of conversational recommender systems. Large language models show promise in simulating human-like behavior, raising the question of their ability to represent a diverse population of users. We introduce a new protocol to measure the degree to which language models can accurately emulate human behavior in conversational recommendation. This protocol is comprised of five tasks, each designed to evaluate a key property that a synthetic user should exhibit: choosing which items to talk about, expressing binary preferences, expressing open-ended preferences, requesting recommendations, and giving feedback. Through evaluation of baseline simulators, we demonstrate these tasks effectively reveal deviations of language models from human behavior, and offer insights on how to reduce the deviations with model selection and prompting strategies.