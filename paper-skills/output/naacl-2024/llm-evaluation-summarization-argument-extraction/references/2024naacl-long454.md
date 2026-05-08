---
title: "Enhancing Argument Summarization: Prioritizing Exhaustiveness in Key Point Generation and Introducing an Automatic Coverage Evaluation Metric"
source: "https://aclanthology.org/2024.naacl-long.454/"
categories: ['llm-evaluation-summarization-argument-extraction', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['argument-summarization', 'key-point-analysis', 'coverage-metric', 'exhaustiveness']
venue: "NAACL 2024"
tldr: "Enhances argument summarization by prioritizing exhaustiveness in key point generation and introducing an automatic coverage metric."
---

# Enhancing Argument Summarization: Prioritizing Exhaustiveness in Key Point Generation and Introducing an Automatic Coverage Evaluation Metric

**Source**: [https://aclanthology.org/2024.naacl-long.454/](https://aclanthology.org/2024.naacl-long.454/)

**TLDR**: Enhances argument summarization by prioritizing exhaustiveness in key point generation and introducing an automatic coverage metric.

## Abstract

AbstractThe proliferation of social media platforms has given rise to the amount of online debates and arguments. Consequently, the need for automatic summarization methods for such debates is imperative, however this area of summarization is rather understudied. The Key Point Analysis (KPA) task formulates argument summarization as representing the summary of a large collection of arguments in the form of concise sentences in bullet-style format, called key points. A sub-task of KPA, called Key Point Generation (KPG), focuses on generating these key points given the arguments. This paper introduces a novel extractive approach for key point generation, that outperforms previous state-of-the-art methods for the task. Our method utilizes an extractive clustering based approach that offers concise, high quality generated key points with higher coverage of reference summaries, and less redundant outputs. In addition, we show that the existing evaluation metrics for summarization such as ROUGE are incapable of differentiating between generated key points of different qualities. To this end, we propose a new evaluation metric for assessing the generated key points by their coverage. Our code can be accessed online.