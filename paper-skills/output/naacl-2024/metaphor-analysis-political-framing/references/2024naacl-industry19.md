---
title: "Reducing hallucination in structured outputs via Retrieval-Augmented Generation"
source: "https://aclanthology.org/2024.naacl-industry.19/"
categories: ['llm-knowledge-reasoning-retrieval', 'metaphor-analysis-political-framing']
tags: ['retrieval-augmented-generation', 'hallucination', 'structured-output']
venue: "NAACL 2024"
tldr: "Uses Retrieval-Augmented Generation to reduce hallucination in LLMs when producing structured outputs."
---

# Reducing hallucination in structured outputs via Retrieval-Augmented Generation

**Source**: [https://aclanthology.org/2024.naacl-industry.19/](https://aclanthology.org/2024.naacl-industry.19/)

**TLDR**: Uses Retrieval-Augmented Generation to reduce hallucination in LLMs when producing structured outputs.

## Abstract

AbstractA current limitation of Generative AI (GenAI) is its propensity to hallucinate. While Large Language Models (LLM) have taken the world by storm, without eliminating or at least reducing hallucination, real-world GenAI systems will likely continue to face challenges in user adoption. In the process of deploying an enterprise application that produces workflows from natural language requirements, we devised a system leveraging Retrieval-Augmented Generation (RAG) to improve the quality of the structured output that represents such workflows. Thanks to our implementation of RAG, our proposed system significantly reduces hallucination and allows the generalization of our LLM to out-of-domain settings. In addition, we show that using a small, well-trained retriever can reduce the size of the accompanying LLM at no loss in performance, thereby making deployments of LLM-based systems less resource-intensive.