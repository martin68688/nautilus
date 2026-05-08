---
title: "Prompting Few-shot Multi-hop Question Generation via Comprehending Type-aware Semantics"
source: "https://aclanthology.org/2024.findings-naacl.236/"
categories: ['llm-evaluation-summarization-argument-extraction', 'zero-shot-few-shot-multimodal-optimization']
tags: ['question-generation', 'few-shot', 'multi-hop']
venue: "NAACL 2024"
tldr: "Presents a prompting method for few-shot multi-hop question generation that leverages type-aware semantic comprehension of documents."
---

# Prompting Few-shot Multi-hop Question Generation via Comprehending Type-aware Semantics

**Source**: [https://aclanthology.org/2024.findings-naacl.236/](https://aclanthology.org/2024.findings-naacl.236/)

**TLDR**: Presents a prompting method for few-shot multi-hop question generation that leverages type-aware semantic comprehension of documents.

## Abstract

AbstractGiven several documents, multi-hop question generation (MQG) is a task aims to generate complicated questions that require reasoning over multiple pieces of these documents to find the answer. To perform this task, existing studies focus on designing advanced architectures to locate essential keywords or sentences in multiple documents and then generate questions accordingly, where they normally do not note that question types could provide crucial hints for extracting key information from the documents for MQG. In general, supervised approaches are used that rely on large annotated data, which is not available in many low-resource scenarios and thus makes MQG hard in these domains. Consider the recent success of large language models (LLMs) on natural language processing tasks using limited labeled data under few-shot settings, in this paper, we propose an approach named type-aware semantics extraction-based chain-of-thought method (TASE-CoT) for few-shot MQG. Specifically, our approach firstly extracts question types and essential semantic phrases from the given documents and the answer. Then, we design a three-step CoT template to leverage the extracted question type and semantic phrases to predict multi-hop questions. Extensive experiments and the results demonstrate the effectiveness of our approach and the proposed modules.