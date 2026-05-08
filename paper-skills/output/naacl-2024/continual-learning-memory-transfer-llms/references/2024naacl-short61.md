---
title: "Arithmetic Reasoning with LLM: Prolog Generation & Permutation"
source: "https://aclanthology.org/2024.naacl-short.61/"
categories: ['llm-knowledge-reasoning-retrieval', 'continual-learning-memory-transfer-llms']
tags: ['arithmetic-reasoning', 'prolog', 'logic-programming', 'llm']
venue: "NAACL 2024"
tldr: "Improving LLM arithmetic reasoning by generating and permuting Prolog programs to avoid calculation errors."
---

# Arithmetic Reasoning with LLM: Prolog Generation & Permutation

**Source**: [https://aclanthology.org/2024.naacl-short.61/](https://aclanthology.org/2024.naacl-short.61/)

**TLDR**: Improving LLM arithmetic reasoning by generating and permuting Prolog programs to avoid calculation errors.

## Abstract

AbstractInstructing large language models (LLMs) to solve elementary school math problems has shown great success using Chain of Thought (CoT). However, the CoT approach relies on an LLM to generate a sequence of arithmetic calculations which can be prone to cascaded calculation errors. We hypothesize that an LLM should focus on extracting predicates and generating symbolic formulas from the math problem description so that the underlying calculation can be done via an external code interpreter. We investigate using LLM to generate Prolog programs to solve mathematical questions. Experimental results show that our Prolog-based arithmetic problem-solving outperforms CoT generation in the GSM8K benchmark across three distinct LLMs. In addition, given the insensitive ordering of predicates and symbolic formulas in Prolog, we propose to permute the ground truth predicates for more robust LLM training via data augmentation.