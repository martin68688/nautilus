---
title: "Multi-Granularity Guided Fusion-in-Decoder"
source: "https://aclanthology.org/2024.findings-naacl.142/"
categories: ['llm-knowledge-reasoning-retrieval']
tags: ['open-domain-qa', 'fusion-in-decoder', 'multi-granularity', 'retrieval']
venue: "NAACL 2024"
tldr: "Improves Fusion-in-Decoder for open-domain QA by using multi-granularity guidance to better fuse retrieved contexts."
---

# Multi-Granularity Guided Fusion-in-Decoder

**Source**: [https://aclanthology.org/2024.findings-naacl.142/](https://aclanthology.org/2024.findings-naacl.142/)

**TLDR**: Improves Fusion-in-Decoder for open-domain QA by using multi-granularity guidance to better fuse retrieved contexts.

## Abstract

AbstractIn Open-domain Question Answering (ODQA), it is essential to discern relevant contexts as evidence and avoid spurious ones among retrieved results. The model architecture that uses concatenated multiple contexts in the decoding phase, *i.e.*, Fusion-in-Decoder, demonstrates promising performance but generates incorrect outputs from seemingly plausible contexts. To address this problem, we propose the ***M**ulti-**G**ranularity guided **F**usion-**i**n-**D**ecoder (**MGFiD**)*, discerning evidence across multiple levels of granularity. Based on multi-task learning, MGFiD harmonizes passage re-ranking with sentence classification. It aggregates evident sentences into an *anchor vector* that instructs the decoder. Additionally, it improves decoding efficiency by reusing the results of passage re-ranking for *passage pruning*. Through our experiments, MGFiD outperforms existing models on the Natural Questions (NQ) and TriviaQA (TQA) datasets, highlighting the benefits of its multi-granularity solution.