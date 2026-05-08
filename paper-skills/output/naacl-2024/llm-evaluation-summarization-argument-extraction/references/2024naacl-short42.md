---
title: "Control-DAG: Constrained Decoding for Non-Autoregressive Directed Acyclic T5 using Weighted Finite State Automata"
source: "https://aclanthology.org/2024.naacl-short.42/"
categories: ['efficient-transformer-optimization-techniques', 'llm-evaluation-summarization-argument-extraction']
tags: ['non-autoregressive', 'constrained-decoding', 'generation']
venue: "NAACL 2024"
tldr: "Applies constrained decoding with finite state automata to improve a non-autoregressive model for NLG tasks."
---

# Control-DAG: Constrained Decoding for Non-Autoregressive Directed Acyclic T5 using Weighted Finite State Automata

**Source**: [https://aclanthology.org/2024.naacl-short.42/](https://aclanthology.org/2024.naacl-short.42/)

**TLDR**: Applies constrained decoding with finite state automata to improve a non-autoregressive model for NLG tasks.

## Abstract

AbstractThe Directed Acyclic Transformer is a fast non-autoregressive (NAR) model that performs well in Neural Machine Translation. Two issues prevent its application to general Natural Language Generation (NLG) tasks: frequent Out-Of-Vocabulary (OOV) errors and the inability to faithfully generate entity names. We introduce Control-DAG, a constrained decoding algorithm for our Directed Acyclic T5 (DA-T5) model which offers lexical, vocabulary and length control. We show that Control-DAG significantly enhances DA-T5 on the Schema Guided Dialogue and the DART datasets, establishing strong NAR results for Task-Oriented Dialogue and Data-to-Text NLG.