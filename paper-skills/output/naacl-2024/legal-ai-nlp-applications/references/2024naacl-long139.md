---
title: "Efficient Benchmarking (of Language Models)"
source: "https://aclanthology.org/2024.naacl-long.139/"
categories: ['legal-ai-nlp-applications', 'language-model-evaluation-benchmarks']
tags: ['benchmarking', 'efficiency', 'model-evaluation']
venue: "NAACL 2024"
tldr: "Addresses the massive computational cost of comprehensive LM benchmarking and proposes methods for efficient evaluation."
---

# Efficient Benchmarking (of Language Models)

**Source**: [https://aclanthology.org/2024.naacl-long.139/](https://aclanthology.org/2024.naacl-long.139/)

**TLDR**: Addresses the massive computational cost of comprehensive LM benchmarking and proposes methods for efficient evaluation.

## Abstract

AbstractThe increasing versatility of language models (LMs) has given rise to a new class of benchmarks that comprehensively assess a broad range of capabilities. Such benchmarks are associated with massive computational costs, extending to thousands of GPU hours per model. However, the efficiency aspect of these evaluation efforts had raised little discussion in the literature.In this work, we present the problem of Efficient Benchmarking, namely, intelligently reducing the computation costs of LM evaluation without compromising reliability. Using the HELM benchmark as a test case, we investigate how different benchmark design choices affect the computation-reliability trade-off. We propose to evaluate the reliability of such decisions, by using a new measure – Decision Impact on Reliability, DIoR for short.We find, for example, that a benchmark leader may change by merely removing a low-ranked model from the benchmark, and observe that a correct benchmark ranking can be obtained by considering only a fraction of the evaluation examples.Based on our findings, we outline a set of concrete recommendations for efficient benchmark design and utilization practices. To take a step further, we use our findings to propose an evaluation algorithm, that, when applied to the HELM benchmark, leads to dramatic cost savings with minimal loss of benchmark reliability, often reducing computation by x100 or more.