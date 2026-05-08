---
title: "Extractive Summarization with Text Generator"
source: "https://aclanthology.org/2024.naacl-long.9/"
categories: ['llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['summarization', 'extractive', 'text-generation']
venue: "NAACL 2024"
tldr: "Proposes an extractive summarization method using a text generator to avoid reliance on imperfect pseudo-labels."
---

# Extractive Summarization with Text Generator

**Source**: [https://aclanthology.org/2024.naacl-long.9/](https://aclanthology.org/2024.naacl-long.9/)

**TLDR**: Proposes an extractive summarization method using a text generator to avoid reliance on imperfect pseudo-labels.

## Abstract

AbstractStandard extractive systems suffer from the lack of gold training signals since existing corpora solely provide document and human-written summary pairs while disregarding extractive labels. As a result, existing methods resort to imperfect pseudo-labels that are both biased and error-prone, thereby hindering the learning process of extractive models. In contrast, text generators which are commonly employed in abstractive summarization can effortlessly overcome this predicament on account of flexible sequence-to-sequence architectures. Motivated to bypass this inherent limitation, we investigate the possibility of conducting extractive summarization with text generators. Through extensive experiments covering six summarization benchmarks, we show that high-quality extractive summaries can be assembled via approximating the outputs (abstractive summaries) of these generators. Moreover, we find that the approximate summaries correlate positively with the auxiliary summaries (i.e. a better generator enables the production of better extractive summaries). Our results signify a new paradigm for training extractive summarizers i.e. learning with generation (abstractive) objectives rather than extractive schemes.