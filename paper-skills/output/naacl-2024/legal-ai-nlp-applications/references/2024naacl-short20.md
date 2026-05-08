---
title: "Unveiling Divergent Inductive Biases of LLMs on Temporal Data"
source: "https://aclanthology.org/2024.naacl-short.20/"
categories: ['legal-ai-nlp-applications', 'knowledge-conflict-diagnostic-temporal-reasoning', 'logical-reasoning-in-neural-models']
tags: ['temporal-reasoning', 'inductive-bias', 'llm-evaluation']
venue: "NAACL 2024"
tldr: "Unveils divergent inductive biases of LLMs on temporal data."
---

# Unveiling Divergent Inductive Biases of LLMs on Temporal Data

**Source**: [https://aclanthology.org/2024.naacl-short.20/](https://aclanthology.org/2024.naacl-short.20/)

**TLDR**: Unveils divergent inductive biases of LLMs on temporal data.

## Abstract

AbstractUnraveling the intricate details of events in natural language necessitates a subtle understanding of temporal dynamics. Despite the adeptness of Large Language Models (LLMs) in discerning patterns and relationships from data, their inherent comprehension of temporal dynamics remains a formidable challenge. This research meticulously explores these intrinsic challenges within LLMs, with a specific emphasis on evaluating the performance of GPT-3.5 and GPT-4 models in the analysis of temporal data. Employing two distinct prompt types, namely Question Answering (QA) format and Textual Entailment (TE) format, our analysis probes into both implicit and explicit events. The findings underscore noteworthy trends, revealing disparities in the performance of GPT-3.5 and GPT-4. Notably, biases toward specific temporal relationships come to light, with GPT-3.5 demonstrating a preference for “AFTER” in the QA format for both implicit and explicit events, while GPT-4 leans towards “BEFORE”. Furthermore, a consistent pattern surfaces wherein GPT-3.5 tends towards “TRUE”, and GPT-4 exhibits a preference for “FALSE” in the TE format for both implicit and explicit events. This persistent discrepancy between GPT-3.5 and GPT-4 in handling temporal data highlights the intricate nature of inductive bias in LLMs, suggesting that the evolution of these models may not merely mitigate bias but may introduce new layers of complexity.