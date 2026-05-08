---
title: "Mitigating Hallucination in Abstractive Summarization with Domain-Conditional Mutual Information"
source: "https://aclanthology.org/2024.findings-naacl.117/"
categories: ['clinical-nlp-biomedical-applications', 'llm-evaluation-summarization-argument-extraction']
tags: ['abstractive-summarization', 'hallucination-mitigation', 'domain-conditioning']
venue: "NAACL 2024"
tldr: "Domain-conditional mutual information is used to mitigate hallucination in abstractive summarization by reducing off-topic text generation."
---

# Mitigating Hallucination in Abstractive Summarization with Domain-Conditional Mutual Information

**Source**: [https://aclanthology.org/2024.findings-naacl.117/](https://aclanthology.org/2024.findings-naacl.117/)

**TLDR**: Domain-conditional mutual information is used to mitigate hallucination in abstractive summarization by reducing off-topic text generation.

## Abstract

AbstractA primary challenge in abstractive summarization is hallucination—the phenomenon where a model generates plausible text that is absent in the source text. We hypothesize that the domain (or topic) of the source text triggers the model to generate text that is highly probable in the domain, neglecting the details of the source text. To alleviate this model bias, we introduce a decoding strategy based on domain-conditional pointwise mutual information. This strategy adjusts the generation probability of each token by comparing it with the token’s marginal probability within the domain of the source text. According to evaluation on the XSUM dataset, our method demonstrates improvement in terms of faithfulness and source relevance.