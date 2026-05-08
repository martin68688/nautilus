---
title: "Bridging the Gap between Different Vocabularies for LLM Ensemble"
source: "https://aclanthology.org/2024.naacl-long.395/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['efficient-training', 'model-ensemble', 'vocabulary']
venue: "NAACL 2024"
tldr: "Proposes a method to bridge vocabulary gaps between different LLMs to enable effective ensembling and harness complementary strengths."
---

# Bridging the Gap between Different Vocabularies for LLM Ensemble

**Source**: [https://aclanthology.org/2024.naacl-long.395/](https://aclanthology.org/2024.naacl-long.395/)

**TLDR**: Proposes a method to bridge vocabulary gaps between different LLMs to enable effective ensembling and harness complementary strengths.

## Abstract

AbstractEnsembling different large language models (LLMs) to unleash their complementary potential and harness their individual strengths is highly valuable. Nevertheless, vocabulary discrepancies among various LLMs have constrained previous studies to either selecting or blending completely generated outputs. This limitation hinders the dynamic correction and enhancement of outputs during the generation process, resulting in a limited capacity for effective ensemble. To address this issue, we propose a novel method to Ensemble LLMs via Vocabulary Alignment (EVA). EVA bridges the lexical gap among various LLMs, enabling meticulous ensemble at each generation step. Specifically, we first learn mappings between the vocabularies of different LLMs with the assistance of overlapping tokens. Subsequently, these mappings are employed to project output distributions of LLMs into a unified space, facilitating a fine-grained ensemble. Finally, we design a filtering strategy to exclude models that generate unfaithful tokens. Experimental results on commonsense reasoning, arithmetic reasoning, machine translation, and data-to-text generation tasks demonstrate the superiority of our approach compared with individual LLMs and previous ensemble methods conducted on complete outputs. Further analyses confirm that our approach can leverage knowledge from different language models and yield consistent improvement.