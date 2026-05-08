---
title: "DriftWatch: A Tool that Automatically Detects Data Drift and Extracts Representative Examples Affected by Drift"
source: "https://aclanthology.org/2024.naacl-industry.28/"
categories: ['code-search-clone-detection', 'bayes-risk-decoding-and-data-drift']
tags: ['data-drift', 'monitoring-tool', 'example-extraction', 'ml-ops']
venue: "NAACL 2024"
tldr: "Presents a tool that automatically detects data drift in ML models and extracts representative examples affected by it."
---

# DriftWatch: A Tool that Automatically Detects Data Drift and Extracts Representative Examples Affected by Drift

**Source**: [https://aclanthology.org/2024.naacl-industry.28/](https://aclanthology.org/2024.naacl-industry.28/)

**TLDR**: Presents a tool that automatically detects data drift in ML models and extracts representative examples affected by it.

## Abstract

AbstractData drift, which denotes a misalignment between the distribution of reference (i.e., training) and production data, constitutes a significant challenge for AI applications, as it undermines the generalisation capacity of machine learning (ML) models. Therefore, it is imperative to proactively identify data drift before users meet with performance degradation. Moreover, to ensure the successful execution of AI services, endeavours should be directed not only toward detecting the occurrence of drift but also toward effectively addressing this challenge. % considering the limited resources prevalent in practical industrial domains. In this work, we introduce a tool designed to detect data drift in text data. In addition, we propose an unsupervised sampling technique for extracting representative examples from drifted instances. This approach bestows a practical advantage by significantly reducing expenses associated with annotating the labels for drifted instances, an essential prerequisite for retraining the model to sustain its performance on production data.