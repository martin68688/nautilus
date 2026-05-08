---
title: "BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models"
source: "https://aclanthology.org/2024.findings-naacl.155/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-ranking-benchmarking-adaptation']
tags: ['knowledge-probing', 'relational-knowledge', 'causal-models']
venue: "NAACL 2024"
tldr: "Introduces BEAR, a unified framework for evaluating relational knowledge in causal and masked language models."
---

# BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.155/](https://aclanthology.org/2024.findings-naacl.155/)

**TLDR**: Introduces BEAR, a unified framework for evaluating relational knowledge in causal and masked language models.

## Abstract

AbstractKnowledge probing assesses to which degree a language model (LM) has successfully learned relational knowledge during pre-training. Probing is an inexpensive way to compare LMs of different sizes and training configurations. However, previous approaches rely on the objective function used in pre-training LMs and are thus applicable only to masked or causal LMs. As a result, comparing different types of LMs becomes impossible. To address this, we propose an approach that uses an LM’s inherent ability to estimate the log-likelihood of any given textual statement. We carefully design an evaluation dataset of 7,731 instances (40,916 in a larger variant) from which we produce alternative statements for each relational fact, one of which is correct. We then evaluate whether an LM correctly assigns the highest log-likelihood to the correct statement. Our experimental evaluation of 22 common LMs shows that our proposed framework, BEAR, can effectively probe for knowledge across different LM types. We release the BEAR datasets and an open-source framework that implements the probing approach to the research community to facilitate the evaluation and development of LMs.