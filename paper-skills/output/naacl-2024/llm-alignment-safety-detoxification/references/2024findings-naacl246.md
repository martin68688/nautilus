---
title: "Sentiment Analysis in the Era of Large Language Models: A Reality Check"
source: "https://aclanthology.org/2024.findings-naacl.246/"
categories: ['llm-alignment-safety-detoxification', 'large-language-model-evaluation-augmentation']
tags: ['sentiment_analysis', 'LLM_evaluation', 'reality_check']
venue: "NAACL 2024"
tldr: "A reality check on employing large language models for various sentiment analysis tasks and challenges."
---

# Sentiment Analysis in the Era of Large Language Models: A Reality Check

**Source**: [https://aclanthology.org/2024.findings-naacl.246/](https://aclanthology.org/2024.findings-naacl.246/)

**TLDR**: A reality check on employing large language models for various sentiment analysis tasks and challenges.

## Abstract

AbstractSentiment analysis (SA) has been a long-standing research area in natural language processing. With the recent advent of large language models (LLMs), there is great potential for their employment on SA problems. However, the extent to which current LLMs can be leveraged for different sentiment analysis tasks remains unclear. This paper aims to provide a comprehensive investigation into the capabilities of LLMs in performing various sentiment analysis tasks, from conventional sentiment classification to aspect-based sentiment analysis and multifaceted analysis of subjective texts. We evaluate performance across 13 tasks on 26 datasets and compare the results against small language models (SLMs) trained on domain-specific datasets. Our study reveals that while LLMs demonstrate satisfactory performance in simpler tasks, they lag behind in more complex tasks requiring a deeper understanding of specific sentiment phenomena or structured sentiment information. However, LLMs significantly outperform SLMs in few-shot learning settings, suggesting their potential when annotation resources are limited. We also highlight the limitations of current evaluation practices in assessing LLMs’ SA abilities and propose a novel benchmark, SentiEval, for a more comprehensive and realistic evaluation. Data and code are available at https://github.com/DAMO-NLP-SG/LLM-Sentiment.