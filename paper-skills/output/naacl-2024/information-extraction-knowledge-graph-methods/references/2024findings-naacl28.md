---
title: "Leveraging Contextual Information for Effective Entity Salience Detection"
source: "https://aclanthology.org/2024.findings-naacl.28/"
categories: ['information-extraction-knowledge-graph-methods']
tags: ['entity-salience', 'information-extraction', 'news-articles']
venue: "NAACL 2024"
tldr: "A method for detecting salient entities in documents by leveraging contextual information."
---

# Leveraging Contextual Information for Effective Entity Salience Detection

**Source**: [https://aclanthology.org/2024.findings-naacl.28/](https://aclanthology.org/2024.findings-naacl.28/)

**TLDR**: A method for detecting salient entities in documents by leveraging contextual information.

## Abstract

AbstractIn text documents such as news articles, the content and key events usually revolve around a subset of all the entities mentioned in a document. These entities, often deemed as salient entities, provide useful cues of the aboutness of a document to a reader. Identifying the salience of entities was found helpful in several downstream applications such as search, ranking, and entity-centric summarization, among others. Prior work on salient entity detection mainly focused on machine learning models that require heavy feature engineering. We show that fine-tuning medium-sized language models with a cross-encoder style architecture yields substantial performance gains over feature engineering approaches. To this end, we conduct a comprehensive benchmarking of four publicly available datasets using models representative of the medium-sized pre-trained language model family. Additionally, we show that zero-shot prompting of instruction-tuned language models yields inferior results, indicating the task’s uniqueness and complexity.