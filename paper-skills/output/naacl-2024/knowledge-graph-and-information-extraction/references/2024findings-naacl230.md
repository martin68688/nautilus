---
title: "FIRE: A Dataset for Financial Relation Extraction"
source: "https://aclanthology.org/2024.findings-naacl.230/"
categories: ['knowledge-graph-and-information-extraction']
tags: ['relation-extraction', 'financial', 'dataset']
venue: "NAACL 2024"
tldr: "Introducing FIRE, a dataset for financial named entity and relation extraction."
---

# FIRE: A Dataset for Financial Relation Extraction

**Source**: [https://aclanthology.org/2024.findings-naacl.230/](https://aclanthology.org/2024.findings-naacl.230/)

**TLDR**: Introducing FIRE, a dataset for financial named entity and relation extraction.

## Abstract

AbstractThis paper introduces FIRE (**FI**nancial **R**elation **E**xtraction), a sentence-level dataset of named entities and relations within the financial sector. Comprising 3,025 instances, the dataset encapsulates 13 named entity types along with 18 relation types. Sourced from public financial reports and financial news articles, FIRE captures a wide array of financial information about a business including, but not limited to, corporate structure, business model, revenue streams, and market activities such as acquisitions. The full dataset was labeled by a single annotator to minimize labeling noise. The labeling time for each sentence was recorded during the labeling process. We show how this feature, along with curriculum learning techniques, can be used to improved a model’s performance. The FIRE dataset is designed to serve as a valuable resource for training and evaluating machine learning algorithms in the domain of financial information extraction. The dataset and the code to reproduce our experimental results are available at https://github.com/hmhamad/FIRE. The repository for the labeling tool can be found at https://github.com/abhinav-kumar-thakur/relation-extraction-annotator.