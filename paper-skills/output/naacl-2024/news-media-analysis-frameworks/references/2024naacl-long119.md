---
title: "P3Sum: Preserving Author’s Perspective in News Summarization with Diffusion Language Models"
source: "https://aclanthology.org/2024.naacl-long.119/"
categories: ['llm-evaluation-summarization-argument-extraction', 'news-media-analysis-frameworks']
tags: ['summarization', 'political-bias', 'diffusion-models']
venue: "NAACL 2024"
tldr: "Uses diffusion language models to preserve the author's political perspective in news summarization."
---

# P3Sum: Preserving Author’s Perspective in News Summarization with Diffusion Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.119/](https://aclanthology.org/2024.naacl-long.119/)

**TLDR**: Uses diffusion language models to preserve the author's political perspective in news summarization.

## Abstract

AbstractIn this work, we take a first step towards designing summarization systems that are faithful to the author’s intent, not only the semantic content of the article. Focusing on a case study of preserving political perspectives in news summarization, we find that existing approaches alter the political opinions and stances of news articles in more than 50% of summaries, misrepresenting the intent and perspectives of the news authors. We thus propose P3Sum, a diffusion model-based summarization approach controlled by political perspective classifiers. In P3Sum, the political leaning of a generated summary is iteratively evaluated at each decoding step, and any drift from the article’s original stance incurs a loss back-propagated to the embedding layers, steering the political stance of the summary at inference time. Extensive experiments on three news summarization datasets demonstrate that P3Sum outperforms state-of-the-art summarization systems and large language models by up to 13.7% in terms of the success rate of stance preservation, with competitive performance on standard metrics of summarization quality. Our findings present a first analysis of preservation of pragmatic features in summarization, highlight the lacunae in existing summarization models—that even state-of-the-art models often struggle to preserve author’s intents—and develop new summarization systems that are more faithful to author’s perspectives.