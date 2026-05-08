---
title: "MCECR: A Novel Dataset for Multilingual Cross-Document Event Coreference Resolution"
source: "https://aclanthology.org/2024.findings-naacl.245/"
categories: ['large-language-model-evaluation-augmentation', 'knowledge-graph-and-information-extraction']
tags: ['event-coreference', 'multilingual', 'cross-document', 'dataset']
venue: "NAACL 2024"
tldr: "Introduces a multilingual dataset for cross-document event coreference resolution."
---

# MCECR: A Novel Dataset for Multilingual Cross-Document Event Coreference Resolution

**Source**: [https://aclanthology.org/2024.findings-naacl.245/](https://aclanthology.org/2024.findings-naacl.245/)

**TLDR**: Introduces a multilingual dataset for cross-document event coreference resolution.

## Abstract

AbstractEvent coreference resolution (ECR) is a critical task in information extraction of natural language processing, aiming to identify and link event mentions across multiple documents. Despite recent progress, existing datasets for ECR primarily focus on within-document event coreference and English text, lacking cross-document ECR datasets for multiple languages beyond English. To address this issue, this work presents the first multiligual dataset for cross-document ECR, called MCECR (Multilingual Cross-Document Event Coreference Resolution), that manually annotates a diverse collection of documents for event mentions and coreference in five languages, i.e., English, Spanish, Hindi, Turkish, and Ukrainian. Using sampled articles from Wikinews over various topics as the seeds, our dataset fetches related news articles from the Google search engine to increase the number of non-singleton event clusters. In total, we annotate 5,802 news articles, providing a substantial and varied dataset for multilingual ECR in both within-document and cross-document scenarios. Extensive analysis of the proposed dataset reveals the challenging nature of multilingual event coreference resolution tasks, promoting MCECR as a strong benchmark dataset for future research in this area.