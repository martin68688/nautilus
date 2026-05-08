---
title: "UGIF-DataSet: A New Dataset for Cross-lingual, Cross-modal Sequential actions on the UI"
source: "https://aclanthology.org/2024.findings-naacl.89/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'language-grounded-embodied-navigation']
tags: ['multimodal', 'cross-lingual', 'ui-grounding']
venue: "NAACL 2024"
tldr: "Introduces a dataset for cross-lingual, cross-modal sequential action tasks on smartphone user interfaces."
---

# UGIF-DataSet: A New Dataset for Cross-lingual, Cross-modal Sequential actions on the UI

**Source**: [https://aclanthology.org/2024.findings-naacl.89/](https://aclanthology.org/2024.findings-naacl.89/)

**TLDR**: Introduces a dataset for cross-lingual, cross-modal sequential action tasks on smartphone user interfaces.

## Abstract

AbstractHelp documents are supposed to aid smartphone users in resolving queries such as “How to block calls from unknown numbers?”. However, given a query, identifying the right help document, understanding instructions from the document, and using them to resolve the issue at hand is challenging. The user experience may be enhanced by converting the instructions in the help document to a step-by-step tutorial overlaid on the phone UI. Successful execution of this task requires overcoming research challenges in retrieval, parsing, and grounding in the multilingual-multimodal setting. For example, user queries in one language may have to be matched against instructions in another language, which in turn needs to be grounded in a multimodal UI in yet another language. Moreover, there isn’t any relevant dataset for such a task. In order to bridge this gap, we introduce UGIF-DataSet, a multi-lingual, multi-modal UI grounded dataset for step-by-step task completion on the smartphone, containing 4,184 tasks across 8 languages. The instruction steps in UGIF-DataSet are available only in English, so the challenge involves operations in the cross-modal, cross-lingual setting. We compare the performance of different large language models for this task and find that the end-to-end task completion rate drops from 48% in English to 32% for other languages, demonstrating significant overall headroom for improvement. We are hopeful that UGIF-DataSet and our analysis will aid further research on the important problem of sequential task completion in the multilingual and multimodal setting.