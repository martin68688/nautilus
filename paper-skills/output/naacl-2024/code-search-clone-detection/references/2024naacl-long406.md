---
title: "CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization"
source: "https://aclanthology.org/2024.naacl-long.406/"
categories: ['llm-evaluation-summarization-argument-extraction', 'code-search-clone-detection']
tags: ['summarization', 'dataset', 'news']
venue: "NAACL 2024"
tldr: "Presents a large-scale, high-quality dataset for abstractive news summarization."
---

# CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization

**Source**: [https://aclanthology.org/2024.naacl-long.406/](https://aclanthology.org/2024.naacl-long.406/)

**TLDR**: Presents a large-scale, high-quality dataset for abstractive news summarization.

## Abstract

AbstractTraining a supervised news summarization model requires large amounts of high-quality training data consisting of news articles paired with reference summaries. However, obtaining such data is costly, and existing datasets contain considerable amount of noise. We present a new large-scale and high-quality dataset for supervised abstractive news summarization containing 1.3 million training samples, which we call CCSum. In creating this dataset, we take advantage of the journalistic inverted-pyramid style in news writing: In some articles, the first sentence can be considered a summary of the reported story. Accordingly, among 35 million CommonCrawl News articles, we identify pairs of articles about the same news story and use one article’s first sentence as the summary for the other article. To ensure high quality, we apply strict filters whose parameters we optimize using Bayesian optimization. We show that the resulting dataset is more factual and informative than established summarization datasets; less than 1% of the summaries have major factual inconsistencies with the corresponding news articles, compared to 5.5% to 15.4% in existing datasets, according to our human evaluation. Summarization models trained on our dataset are more favored compared to those trained on CNN/Daily Mail. The proposed dataset can open new opportunities for future research in abstractive summarization.