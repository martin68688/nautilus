---
title: "Ignore Me But Don’t Replace Me: Utilizing Non-Linguistic Elements for Pretraining on the Cybersecurity Domain"
source: "https://aclanthology.org/2024.findings-naacl.3/"
categories: ['efficient-transformer-optimization-techniques', 'llm-backdoor-attacks-defense']
tags: ['domain-adaptation', 'cybersecurity', 'pretraining']
venue: "NAACL 2024"
tldr: "Proposes a pretraining method for cybersecurity NLP that utilizes non-linguistic elements like code and symbols."
---

# Ignore Me But Don’t Replace Me: Utilizing Non-Linguistic Elements for Pretraining on the Cybersecurity Domain

**Source**: [https://aclanthology.org/2024.findings-naacl.3/](https://aclanthology.org/2024.findings-naacl.3/)

**TLDR**: Proposes a pretraining method for cybersecurity NLP that utilizes non-linguistic elements like code and symbols.

## Abstract

AbstractCybersecurity information is often technically complex and relayed through unstructured text, making automation of cyber threat intelligence highly challenging. For such text domains that involve high levels of expertise, pretraining on in-domain corpora has been a popular method for language models to obtain domain expertise. However, cybersecurity texts often contain non-linguistic elements (such as URLs and hash values) that could be unsuitable with the established pretraining methodologies. Previous work in other domains have removed or filtered such text as noise, but the effectiveness of these methods have not been investigated, especially in the cybersecurity domain. We experiment with different pretraining methodologies to account for non-linguistic elements (NLEs) and evaluate their effectiveness through downstream tasks and probing tasks. Our proposed strategy, a combination of selective MLM and jointly training NLE token classification, outperforms the commonly taken approach of replacing NLEs. We use our domain-customized methodology to train CyBERTuned, a cybersecurity domain language model that outperforms other cybersecurity PLMs on most tasks.