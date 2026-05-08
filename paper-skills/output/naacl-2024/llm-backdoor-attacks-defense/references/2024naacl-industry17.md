---
title: "Fighting crime with Transformers: Empirical analysis of address parsing methods in payment data"
source: "https://aclanthology.org/2024.naacl-industry.17/"
categories: ['llm-backdoor-attacks-defense', 'financial-risk-narrative-analysis']
tags: ['address-parsing', 'financial-nlp', 'anti-money-laundering', 'transformers']
venue: "NAACL 2024"
tldr: "Empirically compares transformer-based methods for parsing addresses from payment text to aid anti-money laundering efforts."
---

# Fighting crime with Transformers: Empirical analysis of address parsing methods in payment data

**Source**: [https://aclanthology.org/2024.naacl-industry.17/](https://aclanthology.org/2024.naacl-industry.17/)

**TLDR**: Empirically compares transformer-based methods for parsing addresses from payment text to aid anti-money laundering efforts.

## Abstract

AbstractIn the financial industry, identifying the location of parties involved in payments is a major challenge in the context of Anti-Money Laundering transaction monitoring. For this purpose address parsing entails extracting fields such as street, postal code, or country from free text message attributes. While payment processing platforms are updating their standards with more structured formats such as SWIFT with ISO 20022, address parsing remains essential for a considerable volume of messages. With the emergence of Transformers and Generative Large Language Models (LLM), we explore the performance of state-of-the-art solutions given the constraint of processing a vast amount of daily data. This paper also aims to show the need for training robust models capable of dealing with real-world noisy transactional data. Our results suggest that a well fine-tuned Transformer model using early-stopping significantly outperforms other approaches. Nevertheless, generative LLMs demonstrate strong zero_shot performance and warrant further investigations.