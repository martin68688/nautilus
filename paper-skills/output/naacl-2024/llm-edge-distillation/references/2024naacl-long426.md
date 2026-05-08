---
title: "Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval"
source: "https://aclanthology.org/2024.naacl-long.426/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-edge-distillation']
tags: ['code-generation', 'ui', 'automated-feedback']
venue: "NAACL 2024"
tldr: "Finetuning LLMs to generate user interface code using automated feedback from compilers and visual similarity metrics."
---

# Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval

**Source**: [https://aclanthology.org/2024.naacl-long.426/](https://aclanthology.org/2024.naacl-long.426/)

**TLDR**: Finetuning LLMs to generate user interface code using automated feedback from compilers and visual similarity metrics.

## Abstract

AbstractThere has been limited success for dense retrieval models in multilingual retrieval, due to uneven and scarce training data available across multiple languages. Synthetic training data generation is promising (e.g., InPars or Promptagator), but has been investigated only for English. Therefore, to study model capabilities across both cross-lingual and monolingual retrieval tasks, we develop **SWIM-IR**, a synthetic retrieval training dataset containing 33 (high to very-low resource) languages for fine-tuning multilingual dense retrievers without requiring any human supervision. To construct SWIM-IR, we propose SAP (summarize-then-ask prompting), where the large language model (LLM) generates a textual summary prior to the query generation step. SAP assists the LLM in generating informative queries in the target language. Using SWIM-IR, we explore synthetic fine-tuning of multilingual dense retrieval models and evaluate them robustly on three retrieval benchmarks: XOR-Retrieve (cross-lingual), MIRACL (monolingual) and XTREME-UP (cross-lingual). Our models, called SWIM-X, are competitive with human-supervised dense retrieval models, e.g., mContriever-X, finding that SWIM-IR can cheaply substitute for expensive human-labeled retrieval training data. SWIM-IR dataset and SWIM-X models are available at: https://github.com/google-research-datasets/SWIM-IR.