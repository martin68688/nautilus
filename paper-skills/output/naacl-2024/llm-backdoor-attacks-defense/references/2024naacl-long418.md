---
title: "Measuring Cross-lingual Transfer in Bytes"
source: "https://aclanthology.org/2024.naacl-long.418/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-backdoor-attacks-defense']
tags: ['cross-lingual-transfer', 'multilingual', 'byte-level']
venue: "NAACL 2024"
tldr: "Measures cross-lingual transfer capability in bytes, comparing multilingual and monolingual models' ability to transfer knowledge."
---

# Measuring Cross-lingual Transfer in Bytes

**Source**: [https://aclanthology.org/2024.naacl-long.418/](https://aclanthology.org/2024.naacl-long.418/)

**TLDR**: Measures cross-lingual transfer capability in bytes, comparing multilingual and monolingual models' ability to transfer knowledge.

## Abstract

AbstractMultilingual pretraining has been a successful solution to the challenges posed by the lack of resources for languages. These models can transfer knowledge to target languages with minimal or no examples. Recent research suggests that monolingual models also have a similar capability, but the mechanisms behind this transfer remain unclear. Some studies have explored factors like language contamination and syntactic similarity. An emerging line of research suggests that the representations learned by language models contain two components: a language-specific and a language-agnostic component. The latter is responsible for transferring a more universal knowledge. However, there is a lack of comprehensive exploration of these properties across diverse target languages. To investigate this hypothesis, we conducted an experiment inspired by the work on the Scaling Laws for Transfer. We measured the amount of data transferred from a source language to a target language and found that models initialized from diverse languages perform similarly to a target language in a cross-lingual setting. This was surprising because the amount of data transferred to 10 diverse target languages, such as Spanish, Korean, and Finnish, was quite similar. We also found evidence that this transfer is not related to language contamination or language proximity, which strengthens the hypothesis that the model also relies on language-agnostic knowledge. Our experiments have opened up new possibilities for measuring how much data represents the language-agnostic representations learned during pretraining.