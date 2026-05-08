---
title: "SQATIN: Supervised Instruction Tuning Meets Question Answering for Improved Dialogue NLU"
source: "https://aclanthology.org/2024.naacl-long.453/"
categories: ['llm-evaluation-summarization-argument-extraction', 'llm-ranking-benchmarking-adaptation']
tags: ['dialogue', 'instruction-tuning', 'nlu']
venue: "NAACL 2024"
tldr: "Improves dialogue NLU by supervised instruction tuning on a QA dataset derived from dialogue annotations."
---

# SQATIN: Supervised Instruction Tuning Meets Question Answering for Improved Dialogue NLU

**Source**: [https://aclanthology.org/2024.naacl-long.453/](https://aclanthology.org/2024.naacl-long.453/)

**TLDR**: Improves dialogue NLU by supervised instruction tuning on a QA dataset derived from dialogue annotations.

## Abstract

AbstractTask-oriented dialogue (TOD) systems help users execute well-defined tasks across a variety of domains (e.g., flight booking or food ordering), with their Natural Language Understanding (NLU) components being dedicated to the analysis of user utterances, predicting users’ intents (Intent Detection, ID) and extracting values for informational slots (Value Extraction, VE). In most domains, labelled NLU data is scarce, making sample-efficient learning – enabled with effective transfer paradigms – paramount. In this work, we introduce SQATIN, a new framework for dialog NLU based on (i) instruction tuning and (ii) question-answering-based formulation of ID and VE tasks. According to the evaluation on established NLU benchmarks, SQATIN sets the new state of the art in dialogue NLU, substantially surpassing the performance of current models based on standard fine-tuning objectives in both in-domain training and cross-domain transfer, and it also surpasses off-the-shelf large language models for the same task, both in terms of performance and inference efficiency. Furthermore, SQATIN yields particularly large performance gains in cross-domain transfer, owing to the fact that our QA-based instruction tuning leverages similarities between natural language descriptions of classes (i.e., slots and intents) across domains.