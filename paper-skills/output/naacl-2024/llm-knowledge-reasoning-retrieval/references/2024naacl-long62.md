---
title: "Language Models Hallucinate, but May Excel at Fact Verification"
source: "https://aclanthology.org/2024.naacl-long.62/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-alignment-safety-detoxification']
tags: ['reasoning', 'uncertainty', 'program-aided']
venue: "NAACL 2024"
tldr: "Program-aided reasoning with LLMs improves not only accuracy but also the model's ability to know when it is uncertain."
---

# Language Models Hallucinate, but May Excel at Fact Verification

**Source**: [https://aclanthology.org/2024.naacl-long.62/](https://aclanthology.org/2024.naacl-long.62/)

**TLDR**: Program-aided reasoning with LLMs improves not only accuracy but also the model's ability to know when it is uncertain.

## Abstract

AbstractRecent progress in natural language processing (NLP) owes much to remarkable advances in large language models (LLMs). Nevertheless, LLMs frequently “hallucinate,” resulting in non-factual outputs. Our carefully-designed human evaluation substantiates the serious hallucination issue, revealing that even GPT-3.5 produces factual outputs less than 25% of the time. This underscores the importance of fact verifiers in order to measure and incentivize progress. Our systematic investigation affirms that LLMs can be repurposed as effective fact verifiers with strong correlations with human judgments. Surprisingly, FLAN-T5-11B , the least factual generator in our study, performs the best as a fact verifier, even outperforming more capable LLMs like GPT3.5 and ChatGPT. Delving deeper, we analyze the reliance of these LLMs on high-quality evidence, as well as their deficiencies in robustness and generalization ability. Our study presents insights for developing trustworthy generation models.