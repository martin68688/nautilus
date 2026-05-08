---
title: "LLatrieval: LLM-Verified Retrieval for Verifiable Generation"
source: "https://aclanthology.org/2024.naacl-long.305/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-attribution-verification']
tags: ['verifiable-generation', 'retrieval', 'attribution']
venue: "NAACL 2024"
tldr: "Presents a retrieval method for verifiable generation where an LLM verifies document relevance before using them to support its answer."
---

# LLatrieval: LLM-Verified Retrieval for Verifiable Generation

**Source**: [https://aclanthology.org/2024.naacl-long.305/](https://aclanthology.org/2024.naacl-long.305/)

**TLDR**: Presents a retrieval method for verifiable generation where an LLM verifies document relevance before using them to support its answer.

## Abstract

AbstractVerifiable generation aims to let the large language model (LLM) generate text with supporting documents, which enables the user to flexibly verify the answer and makes the LLM’s output more reliable. Retrieval plays a crucial role in verifiable generation. Specifically, the retrieved documents not only supplement knowledge to help the LLM generate correct answers, but also serve as supporting evidence for the user to verify the LLM’s output. However, the widely used retrievers become the bottleneck of the entire pipeline and limit the overall performance. Their capabilities are usually inferior to LLMs since they often have much fewer parameters than the large language model and have not been demonstrated to scale well to the size of LLMs. If the retriever does not correctly find the supporting documents, the LLM can not generate the correct and verifiable answer, which overshadows the LLM’s remarkable abilities. To address these limitations, we propose **LLatrieval** (**L**arge **La**nguage Model Verified Re**trieval**),where the LLM updates the retrieval result until it verifies that the retrieved documents can sufficiently support answering the question. Thus, the LLM can iteratively provide feedback to retrieval and facilitate the retrieval result to fully support verifiable generation. Experiments on ALCE show that LLatrieval significantly outperforms extensive baselines and achieves state-of-the-art results.