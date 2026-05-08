---
title: "GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting"
source: "https://www.semanticscholar.org/paper/34eedbb011e45d80045cadebaf1d01b2ddec22a1"
categories: ['multimodal-time-series-foundation-models']
tags: ['multimodal', 'time-series', 'forecasting', 'llm', 'prompting']
venue: "AAAI 2024"
tldr: "A prompt-based large language model for multimodal time-series forecasting."
---

# GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting

**Source**: [https://www.semanticscholar.org/paper/34eedbb011e45d80045cadebaf1d01b2ddec22a1](https://www.semanticscholar.org/paper/34eedbb011e45d80045cadebaf1d01b2ddec22a1)

**TLDR**: A prompt-based large language model for multimodal time-series forecasting.

## Abstract

Time series forecasting is an essential area of machine learning with a wide range of real-world applications. Most of the previous forecasting models aim to capture dynamic characteristics from uni-modal numerical historical data. Although extra knowledge can boost the time series forecasting performance, it is hard to collect such information. In addition, how to fuse the multimodal information is non-trivial. In this paper, we first propose a general principle of collecting the corresponding textual information from different data sources with the help of modern large language models (LLM). Then, we propose a prompt-based LLM framework to utilize both the numerical data and the textual information simultaneously, named GPT4MTS. In practice, we propose a GDELT-based multimodal time series dataset for news impact forecasting, which provides a concise and well-structured version of time series dataset with textual information for further research in communication. Through extensive experiments, we demonstrate the effectiveness of our proposed method on forecasting tasks with extra-textual information.