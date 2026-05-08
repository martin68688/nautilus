---
title: "Extremely efficient online query encoding for dense retrieval"
source: "https://aclanthology.org/2024.findings-naacl.4/"
categories: ['llm-knowledge-reasoning-retrieval', 'sentiment-emotion-analysis-multimodal-llms', 'fast-exact-nearest-neighbor-retrieval']
tags: ['dense-retrieval', 'query-encoding', 'efficiency']
venue: "NAACL 2024"
tldr: "An extremely efficient online query encoding method reduces latency for dense retrieval by using a simpler model for queries."
---

# Extremely efficient online query encoding for dense retrieval

**Source**: [https://aclanthology.org/2024.findings-naacl.4/](https://aclanthology.org/2024.findings-naacl.4/)

**TLDR**: An extremely efficient online query encoding method reduces latency for dense retrieval by using a simpler model for queries.

## Abstract

AbstractExisting dense retrieval systems utilize the same model architecture for encoding both the passages and the queries, even though queries are much shorter and simpler than passages. This leads to high latency of the query encoding, which is performed online and therefore might impact user experience. We show that combining a standard large passage encoder with a small efficient query encoder can provide significant latency drops with only a small decrease in quality. We offer a pretraining and training solution for multiple small query encoder architectures. Using a small transformer architecture we are able to decrease latency by up to ∼12×, while MRR@10 on the MS MARCO dev set only decreases from 38.2 to 36.2. If this solution does not reach the desired latency requirements, we propose an efficient RNN as the query encoder, which processes the query prefix incrementally and only infers the last word after the query is issued. This shortens latency by ∼38× with only a minor drop in quality, reaching 35.5 MRR@10 score.