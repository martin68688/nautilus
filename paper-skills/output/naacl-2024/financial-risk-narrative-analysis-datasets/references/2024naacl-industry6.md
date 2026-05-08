---
title: "Modeling and Detecting Company Risks from News"
source: "https://aclanthology.org/2024.naacl-industry.6/"
categories: ['financial-risk-narrative-analysis-datasets', 'political-process-nlp-analysis']
tags: ['risk-detection', 'financial-news', 'information-extraction', 'nlp-applications']
venue: "NAACL 2024"
tldr: "A computational framework to automatically extract and categorize company risk factors from news articles using a schema of seven aspects and a multi-stage model."
---

# Modeling and Detecting Company Risks from News

**Source**: [https://aclanthology.org/2024.naacl-industry.6/](https://aclanthology.org/2024.naacl-industry.6/)

**TLDR**: A computational framework to automatically extract and categorize company risk factors from news articles using a schema of seven aspects and a multi-stage model.

## Abstract

AbstractIdentifying risks associated with a company is important to investors and the wellbeing of the overall financial markets. In this study, we build a computational framework to automatically extract company risk factors from news articles. Our newly proposed schema comprises seven distinct aspects, such as supply chain, regulations, and competition. We annotate 666 news articles and benchmark various machine learning models. While large language mod- els have achieved remarkable progress in various types of NLP tasks, our experiment shows that zero-shot and few-shot prompting state-of- the-art LLMs (e.g., Llama-2) can only achieve moderate to low performances in identifying risk factors. In contrast, fine-tuning pre-trained language models yields better results on most risk factors. Using this model, we analyze over 277K Bloomberg News articles and demonstrate that identifying risk factors from news could provide extensive insights into the operations of companies and industries.