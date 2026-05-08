---
title: "Benchmark Transparency: Measuring the Impact of Data on Evaluation"
source: "https://aclanthology.org/2024.naacl-long.86/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'language-model-evaluation-benchmarks']
tags: ['benchmarking', 'data-distribution', 'evaluation']
venue: "NAACL 2024"
tldr: "Proposes a framework to measure the impact of data distribution on NLP model evaluation across multiple dimensions."
---

# Benchmark Transparency: Measuring the Impact of Data on Evaluation

**Source**: [https://aclanthology.org/2024.naacl-long.86/](https://aclanthology.org/2024.naacl-long.86/)

**TLDR**: Proposes a framework to measure the impact of data distribution on NLP model evaluation across multiple dimensions.

## Abstract

AbstractIn this paper we present an exploratory research on quantifying the impact that data distribution has on the performance and evaluation of NLP models. We propose an automated framework that measures the data point distribution across 6 different dimensions: ambiguity, difficulty, discriminability, length, noise, and perplexity.We use disproportional stratified sampling to measure how much the data distribution affects absolute (Acc/F1) and relative (Rank) model performance. We experiment on 2 different datasets (SQUAD and MNLI) and test a total of 135 different models (125 on SQUAD and 10 on MNLI). We demonstrate that without explicit control of the data distribution, standard evaluation frameworks are inconsistent and unreliable. We find that the impact of the data is statistically significant and is often larger than the impact of changing the metric. In a second set of experiments, we demonstrate that the impact of data on evaluation is not just observable, but also predictable. We propose to use benchmark transparency as a method for comparing datasets and quantifying the similarity between them. We find that the “dataset similarity vector” can be used to predict how well a model generalizes out of distribution.