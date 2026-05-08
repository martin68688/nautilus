---
title: "A Hierarchical Network for Multimodal Document-Level Relation Extraction"
source: "https://www.semanticscholar.org/paper/6bb628417bcb94ce6b1edfcaba58825c0fdf5e9c"
categories: ['topological-geometric-data-representation', 'quantum-inspired-ai-optimization-methods']
tags: ['relation-extraction', 'multimodal', 'documents', 'hierarchical-networks']
venue: "AAAI 2024"
tldr: "A hierarchical network for document-level relation extraction that leverages multimodal information to handle long dependencies and mention selection."
---

# A Hierarchical Network for Multimodal Document-Level Relation Extraction

**Source**: [https://www.semanticscholar.org/paper/6bb628417bcb94ce6b1edfcaba58825c0fdf5e9c](https://www.semanticscholar.org/paper/6bb628417bcb94ce6b1edfcaba58825c0fdf5e9c)

**TLDR**: A hierarchical network for document-level relation extraction that leverages multimodal information to handle long dependencies and mention selection.

## Abstract

Document-level relation extraction aims to extract entity relations that span across multiple sentences. This task faces two critical issues: long dependency and mention selection. Prior works address the above problems from the textual perspective, however, it is hard to handle these problems solely based on text information. In this paper, we leverage video information to provide additional evidence for understanding long dependencies and offer a wider perspective for identifying relevant mentions, thus giving rise to a new task named Multimodal Document-level Relation Extraction (MDocRE). To tackle this new task, we construct a human-annotated dataset including documents and relevant videos, which, to the best of our knowledge, is the first document-level relation extraction dataset equipped with video clips. We also propose a hierarchical framework to learn interactions between different dependency levels and a textual-guided transformer architecture that incorporates both textual and video modalities. In addition, we utilize a mention gate module to address the mention-selection problem in both modalities. Experiments on our proposed dataset show that 1) incorporating video information greatly improves model performance; 2) our hierarchical framework has state-of-the-art results compared with both unimodal and multimodal baselines; 3) through collaborating with video information, our model better solves the long-dependency and mention-selection problems.