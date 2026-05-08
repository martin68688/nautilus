---
title: "MOKA: Moral Knowledge Augmentation for Moral Event Extraction"
source: "https://aclanthology.org/2024.naacl-long.252/"
categories: ['knowledge-graph-and-information-extraction', 'llm-ranking-benchmarking-adaptation']
tags: ['moral-reasoning', 'event-extraction', 'knowledge-augmentation']
venue: "NAACL 2024"
tldr: "Augments moral event extraction with external moral knowledge to better identify values in news events."
---

# MOKA: Moral Knowledge Augmentation for Moral Event Extraction

**Source**: [https://aclanthology.org/2024.naacl-long.252/](https://aclanthology.org/2024.naacl-long.252/)

**TLDR**: Augments moral event extraction with external moral knowledge to better identify values in news events.

## Abstract

AbstractNews media often strive to minimize explicit moral language in news articles, yet most articles are dense with moral values as expressed through the reported events themselves. However, values that are reflected in the intricate dynamics among *participating entities* and *moral events* are far more challenging for most NLP systems to detect, including LLMs. To study this phenomenon, we annotate a new dataset, **MORAL EVENTS**, consisting of 5,494 structured event annotations on 474 news articles by diverse US media across the political spectrum. We further propose **MOKA**, a moral event extraction framework with **MO**ral **K**nowledge **A**ugmentation, which leverages knowledge derived from moral words and moral scenarios to produce structural representations of morality-bearing events. Experiments show that **MOKA** outperforms competitive baselines across three moral event understanding tasks. Further analysis shows even ostensibly nonpartisan media engage in the selective reporting of moral events.