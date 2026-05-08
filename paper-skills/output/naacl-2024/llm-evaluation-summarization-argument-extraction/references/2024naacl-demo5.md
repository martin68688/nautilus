---
title: "RTSUM: Relation Triple-based Interpretable Summarization with Multi-level Salience Visualization"
source: "https://aclanthology.org/2024.naacl-demo.5/"
categories: ['llm-evaluation-summarization-argument-extraction', 'knowledge-graph-and-information-extraction']
tags: ['interpretable-summarization', 'relation-triples', 'salience-visualization']
venue: "NAACL 2024"
tldr: "Presents RTSum, an unsupervised summarization framework that uses and visualizes salient relation triples to create interpretable summaries."
---

# RTSUM: Relation Triple-based Interpretable Summarization with Multi-level Salience Visualization

**Source**: [https://aclanthology.org/2024.naacl-demo.5/](https://aclanthology.org/2024.naacl-demo.5/)

**TLDR**: Presents RTSum, an unsupervised summarization framework that uses and visualizes salient relation triples to create interpretable summaries.

## Abstract

AbstractIn this paper, we present RTSum, an unsupervised summarization framework that utilizes relation triples as the basic unit for summarization. Given an input document, RTSum first selects salient relation triples via multi-level salience scoring and then generates a concise summary from the selected relation triples by using a text-to-text language model. On the basis of RTSum, we also develop a web demo for an interpretable summarizing tool, providing fine-grained interpretations with the output summary. With support for customization options, our tool visualizes the salience for textual units at three distinct levels: sentences, relation triples, and phrases. The code, demo, and video are publicly available.