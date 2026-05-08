---
title: "Adapting Fake News Detection to the Era of Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.95/"
categories: ['llm-backdoor-attacks-defense', 'language-model-cultural-linguistic-evaluation']
tags: ['fake-news', 'detection', 'llm-generated-content']
venue: "NAACL 2024"
tldr: "Proposes methods to adapt fake news detection to the era of LLMs, addressing both human-written and machine-generated fake news."
---

# Adapting Fake News Detection to the Era of Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.95/](https://aclanthology.org/2024.findings-naacl.95/)

**TLDR**: Proposes methods to adapt fake news detection to the era of LLMs, addressing both human-written and machine-generated fake news.

## Abstract

AbstractIn the age of large language models (LLMs) and the widespread adoption of AI-driven content creation, the landscape of information dissemination has witnessed a paradigm shift. With the proliferation of both human-written and machine-generated real and fake news, robustly and effectively discerning the veracity of news articles has become an intricate challenge. While substantial research has been dedicated to fake news detection, it has either assumed that all news articles are human-written or has abruptly assumed that all machine-generated news was fake. Thus, a significant gap exists in understanding the interplay between machine-paraphrased real news, machine-generated fake news, human-written fake news, and human-written real news. In this paper, we study this gap by conducting a comprehensive evaluation of fake news detectors trained in various scenarios. Our primary objectives revolve around the following pivotal question: How can we adapt fake news detectors to the era of LLMs?Our experiments reveal an interesting pattern that detectors trained exclusively on human-written articles can indeed perform well at detecting machine-generated fake news, but not vice versa. Moreover, due to the bias of detectors against machine-generated texts (CITATION), they should be trained on datasets with a lower machine-generated news ratio than the test set. Building on our findings, we provide a practical strategy for the development of robust fake news detectors.