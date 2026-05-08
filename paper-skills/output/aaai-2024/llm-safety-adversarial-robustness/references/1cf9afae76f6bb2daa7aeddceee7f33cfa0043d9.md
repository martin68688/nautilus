---
title: "Chain-of-Thought Improves Text Generation with Citations in Large Language Models"
source: "https://www.semanticscholar.org/paper/1cf9afae76f6bb2daa7aeddceee7f33cfa0043d9"
categories: ['reasoning-learning-optimization-in-llms', 'llm-safety-adversarial-robustness']
tags: ['llm', 'reasoning', 'chain-of-thought', 'citation', 'hallucination']
venue: "AAAI 2024"
tldr: "Shows that chain-of-thought prompting improves text generation with citations in LLMs, reducing hallucinations."
---

# Chain-of-Thought Improves Text Generation with Citations in Large Language Models

**Source**: [https://www.semanticscholar.org/paper/1cf9afae76f6bb2daa7aeddceee7f33cfa0043d9](https://www.semanticscholar.org/paper/1cf9afae76f6bb2daa7aeddceee7f33cfa0043d9)

**TLDR**: Shows that chain-of-thought prompting improves text generation with citations in LLMs, reducing hallucinations.

## Abstract

Previous studies disclose that Large Language Models (LLMs) suffer from hallucinations when generating texts, bringing a novel and challenging research topic to the public, which centers on enabling LLMs to generate texts with citations. Existing work exposes two limitations when using LLMs to generate answers to questions with provided documents: unsatisfactory answer correctness and poor citation quality. To tackle the above issues, we investigate using Chain-of-Thought (CoT) to elicit LLMs’ ability to synthesize correct answers from multiple documents, as well as properly cite these documents. Moreover, we propose a Citation Insurance Mechanism, which enables LLMs to detect and cite those missing citations. We conduct experiments on the ALCE benchmark with six open-source LLMs. Experimental results demonstrate that: (1) the CoT prompting strategy significantly improves the quality of text generation with citations; (2) the Citation Insurance Mechanism delivers impressive gains in citation quality at a low cost; (3) our best approach performs comparably as previous best ChatGPT-based baselines. Extensive analyses further validate the effectiveness of the proposed approach.