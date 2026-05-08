---
title: "Hierarchical Attention Graph for Scientific Document Summarization in Global and Local Level"
source: "https://aclanthology.org/2024.findings-naacl.45/"
categories: ['clinical-nlp-biomedical-applications', 'llm-evaluation-summarization-argument-extraction']
tags: ['scientific-summarization', 'graph-attention', 'hierarchical']
venue: "NAACL 2024"
tldr: "A hierarchical attention graph model captures global and local relations for improved scientific document summarization."
---

# Hierarchical Attention Graph for Scientific Document Summarization in Global and Local Level

**Source**: [https://aclanthology.org/2024.findings-naacl.45/](https://aclanthology.org/2024.findings-naacl.45/)

**TLDR**: A hierarchical attention graph model captures global and local relations for improved scientific document summarization.

## Abstract

AbstractScientific document summarization has been a challenging task due to the long structure of the input text. The long input hinders the simultaneous effective modeling of both global high-order relations between sentences and local intra-sentence relations which is the most critical step in extractive summarization. However, existing methods mostly focus on one type of relation, neglecting the simultaneous effective modeling of both relations, which can lead to insufficient learning of semantic representations. In this paper, we propose HAESum, a novel approach utilizing graph neural networks to locally and globally model documents based on their hierarchical discourse structure. First, intra-sentence relations are learned using a local heterogeneous graph. Subsequently, a novel hypergraph self-attention layer is introduced to further enhance the characterization of high-order inter-sentence relations. We validate our approach on two benchmark datasets, and the experimental results demonstrate the effectiveness of HAESum and the importance of considering hierarchical structures in modeling long scientific documents.