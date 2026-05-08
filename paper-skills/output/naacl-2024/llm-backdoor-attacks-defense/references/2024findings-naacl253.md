---
title: "Few-TK: A Dataset for Few-shot Scientific Typed Keyphrase Recognition"
source: "https://aclanthology.org/2024.findings-naacl.253/"
categories: ['topic-modeling-and-keyphrase-generation', 'llm-backdoor-attacks-defense']
tags: ['scientific-text', 'keyphrase', 'information-extraction']
venue: "NAACL 2024"
tldr: "Presents a dataset for few-shot recognition of typed keyphrases in scientific texts."
---

# Few-TK: A Dataset for Few-shot Scientific Typed Keyphrase Recognition

**Source**: [https://aclanthology.org/2024.findings-naacl.253/](https://aclanthology.org/2024.findings-naacl.253/)

**TLDR**: Presents a dataset for few-shot recognition of typed keyphrases in scientific texts.

## Abstract

AbstractScientific texts are distinctive from ordinary texts in quite a few aspects like their vocabulary and discourse structure. Consequently, Information Extraction (IE) tasks for scientific texts come with their own set of challenges. The classical definition of Named Entities restricts the inclusion of all scientific terms under its hood, which is why previous works have used the terms Named Entities and Keyphrases interchangeably. We suggest the rechristening of Named Entities for the scientific domain as Typed Keyphrases (TK), broadening their scope. We advocate for exploring this task in the few-shot domain due to the scarcity of labeled scientific IE data. Currently, no dataset exists for few-shot scientific Typed Keyphrase Recognition. To address this gap, we develop an annotation schema and present Few-TK, a dataset in the AI/ML field that includes scientific Typed Keyphrase annotations on abstracts of 500 research papers. To the best of our knowledge, this is the introductory few-shot Typed Keyphrase recognition dataset and only the second dataset structured specifically for few-shot NER, after Few-NERD. We report the results of several few-shot sequence-labelling models applied to our dataset. The data and code are available at https://github.com/AvishekLahiri/Few_TK.git