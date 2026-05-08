---
title: "Improving Health Question Answering with Reliable and Time-Aware Evidence Retrieval"
source: "https://aclanthology.org/2024.findings-naacl.295/"
categories: ['llm-knowledge-reasoning-retrieval', 'clinical-nlp-biomedical-applications']
tags: ['health-qa', 'evidence-retrieval', 'reliability', 'temporal']
venue: "NAACL 2024"
tldr: "Improves health QA with reliable and time-aware evidence retrieval from the web."
---

# Improving Health Question Answering with Reliable and Time-Aware Evidence Retrieval

**Source**: [https://aclanthology.org/2024.findings-naacl.295/](https://aclanthology.org/2024.findings-naacl.295/)

**TLDR**: Improves health QA with reliable and time-aware evidence retrieval from the web.

## Abstract

AbstractIn today’s digital world, seeking answers to health questions on the Internet is a common practice. However, existing question answering (QA) systems often rely on using pre-selected and annotated evidence documents, thus making them inadequate for addressing novel questions. Our study focuses on the open-domain QA setting, where the key challenge is to first uncover relevant evidence in large knowledge bases. By utilizing the common retrieve-then-read QA pipeline and PubMed as a trustworthy collection of medical research documents, we answer health questions from three diverse datasets. We modify different retrieval settings to observe their influence on the QA pipeline’s performance, including the number of retrieved documents, sentence selection process, the publication year of articles, and their number of citations. Our results reveal that cutting down on the amount of retrieved documents and favoring more recent and highly cited documents can improve the final macro F1 score up to 10%. We discuss the results, highlight interesting examples, and outline challenges for future research, like managing evidence disagreement and crafting user-friendly explanations.