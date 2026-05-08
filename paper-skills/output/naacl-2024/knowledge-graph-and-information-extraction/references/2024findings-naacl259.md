---
title: "Natural Language Embedded Programs for Hybrid Language Symbolic Reasoning"
source: "https://aclanthology.org/2024.findings-naacl.259/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction', 'logical-reasoning-in-neural-models']
tags: ['symbolic-reasoning', 'program-synthesis', 'math']
venue: "NAACL 2024"
tldr: "Proposes Natural Language Embedded Programs, a framework for hybrid language-symbolic reasoning over natural language."
---

# Natural Language Embedded Programs for Hybrid Language Symbolic Reasoning

**Source**: [https://aclanthology.org/2024.findings-naacl.259/](https://aclanthology.org/2024.findings-naacl.259/)

**TLDR**: Proposes Natural Language Embedded Programs, a framework for hybrid language-symbolic reasoning over natural language.

## Abstract

AbstractHow can we perform computations over natural language representations to solve tasks that require symbolic and numeric reasoning? We propose natural language embedded programs (NLEP) as a unifying framework for addressing math/symbolic reasoning, natural language understanding, and instruction following tasks. Our approach prompts a language model to generate full Python programs that define functions over data structures which contain natural language representations of structured knowledge. A Python interpreter then executes the generated code and prints the output. Despite using a task-general prompt, we find that this approach can improve upon strong baselines across a range of different tasks including math and symbolic reasoning, text classification, question answering, and instruction following. We found that the generated programs are interpretable since they outline the exact reasoning process followed by the program interpreter.