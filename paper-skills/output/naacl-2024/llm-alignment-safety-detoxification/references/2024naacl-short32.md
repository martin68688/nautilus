---
title: "LLM-Driven Knowledge Injection Advances Zero-Shot and Cross-Target Stance Detection"
source: "https://aclanthology.org/2024.naacl-short.32/"
categories: ['llm-alignment-safety-detoxification']
tags: ['stance-detection', 'knowledge-injection', 'zero-shot']
venue: "NAACL 2024"
tldr: "Uses LLMs to inject knowledge for advancing zero-shot and cross-target stance detection."
---

# LLM-Driven Knowledge Injection Advances Zero-Shot and Cross-Target Stance Detection

**Source**: [https://aclanthology.org/2024.naacl-short.32/](https://aclanthology.org/2024.naacl-short.32/)

**TLDR**: Uses LLMs to inject knowledge for advancing zero-shot and cross-target stance detection.

## Abstract

AbstractStance detection aims at inferring an author’s attitude towards a specific target in a text. Prior methods mainly consider target-related background information for a better understanding of targets while neglecting the accompanying input texts. In this study, we propose to prompt Large Language Models (LLMs) to explicitly extract the relationship between paired text and target as contextual knowledge. We then inject such LLM-driven knowledge into a generation model BART to exploit the rich contexts and semantics. Moreover, to further enhance the decoding capability of BART, a novel prototypical contrastive scheme is designed to align input contents with stance labels. Our experimental results demonstrate the state-of-the-art performance across several publicly available datasets, showcasing effectiveness in both zero-shot and cross-target stance detection scenarios. We publicly release our code to facilitate future research.