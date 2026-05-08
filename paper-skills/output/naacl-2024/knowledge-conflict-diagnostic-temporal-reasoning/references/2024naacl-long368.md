---
title: "Separation and Fusion: A Novel Multiple Token Linking Model for Event Argument Extraction"
source: "https://aclanthology.org/2024.naacl-long.368/"
categories: ['knowledge-graph-and-information-extraction', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['event-argument-extraction', 'token-linking', 'information-extraction']
venue: "NAACL 2024"
tldr: "Introduces a novel multiple token linking model for event argument extraction that separates and fuses argument roles."
---

# Separation and Fusion: A Novel Multiple Token Linking Model for Event Argument Extraction

**Source**: [https://aclanthology.org/2024.naacl-long.368/](https://aclanthology.org/2024.naacl-long.368/)

**TLDR**: Introduces a novel multiple token linking model for event argument extraction that separates and fuses argument roles.

## Abstract

AbstractIn event argument extraction (EAE), a promising approach involves jointly encoding text and argument roles, and performing multiple token linking operations. This approach further falls into two categories. One extracts arguments within a single event, while the other attempts to extract arguments from multiple events simultaneously. However, the former lacks to leverage cross-event information and the latter requires tougher predictions with longer encoded role sequences and extra linking operations. In this paper, we design a novel separation-and-fusion paradigm to separately acquire cross-event information and fuse it into the argument extraction of a target event. Following the paradigm, we propose a novel multiple token linking model named Sep2F, which can effectively build event correlations via roles and preserve the simple linking predictions of single-event extraction. In particular, we employ one linking module to extract arguments for the target event and another to aggregate the role information of multiple events. More importantly, we propose a novel two-fold fusion module to ensure that the aggregated cross-event information serves EAE well. We evaluate our proposed model on sentence-level and document-level datasets, including ACE05, RAMS, WikiEvents and MLEE. The extensive experimental results indicate that our model outperforms the state-of-the-art EAE models on all the datasets.