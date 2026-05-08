---
title: "REXEL: An End-to-end Model for Document-Level Relation Extraction and Entity Linking"
source: "https://aclanthology.org/2024.naacl-industry.11/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['relation-extraction', 'entity-linking', 'end-to-end', 'document-level']
venue: "NAACL 2024"
tldr: "An end-to-end model jointly performs document-level relation extraction and entity linking, overcoming limitations of pipeline approaches."
---

# REXEL: An End-to-end Model for Document-Level Relation Extraction and Entity Linking

**Source**: [https://aclanthology.org/2024.naacl-industry.11/](https://aclanthology.org/2024.naacl-industry.11/)

**TLDR**: An end-to-end model jointly performs document-level relation extraction and entity linking, overcoming limitations of pipeline approaches.

## Abstract

AbstractExtracting structured information from unstructured text is critical for many downstream NLP applications and is traditionally achieved by closed information extraction (cIE). However, existing approaches for cIE suffer from two limitations: (i) they are often pipelines which makes them prone to error propagation, and/or (ii) they are restricted to sentence level which prevents them from capturing long-range dependencies and results in expensive inference time. We address these limitations by proposing REXEL, a highly efficient and accurate model for the joint task of document level cIE (DocIE). REXEL performs mention detection, entity typing, entity disambiguation, coreference resolution and document-level relation classification in a single forward pass to yield facts fully linked to a reference knowledge graph. It is on average 11 times faster than competitive existing approaches in a similar setting and performs competitively both when optimised for any of the individual sub-task and a variety of combinations of different joint tasks, surpassing the baselines by an average of more than 6 F1 points. The combination of speed and accuracy makes REXEL an accurate cost-efficient system for extracting structured information at web-scale. We also release an extension of the DocRED dataset to enable benchmarking of future work on DocIE, which will be available at https://github.com/amazon-science/e2e-docie.