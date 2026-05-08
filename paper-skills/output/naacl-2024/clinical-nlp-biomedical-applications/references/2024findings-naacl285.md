---
title: "CARE: Extracting Experimental Findings From Clinical Literature"
source: "https://aclanthology.org/2024.findings-naacl.285/"
categories: ['clinical-nlp-biomedical-applications', 'knowledge-graph-and-information-extraction']
tags: ['clinical-information-extraction', 'biomedical-nlp', 'scientific-literature']
venue: "NAACL 2024"
tldr: "Focuses on extracting fine-grained experimental findings from clinical literature to capture real-world complexity."
---

# CARE: Extracting Experimental Findings From Clinical Literature

**Source**: [https://aclanthology.org/2024.findings-naacl.285/](https://aclanthology.org/2024.findings-naacl.285/)

**TLDR**: Focuses on extracting fine-grained experimental findings from clinical literature to capture real-world complexity.

## Abstract

AbstractExtracting fine-grained experimental findings from literature can provide dramatic utility for scientific applications. Prior work has developed annotation schemas and datasets for limited aspects of this problem, failing to capture the real-world complexity and nuance required. Focusing on biomedicine, this work presents CARE—a new IE dataset for the task of extracting clinical findings. We develop a new annotation schema capturing fine-grained findings as n-ary relations between entities and attributes, which unifies phenomena challenging for current IE systems such as discontinuous entity spans, nested relations, variable arity n-ary relations and numeric results in a single schema. We collect extensive annotations for 700 abstracts from two sources: clinical trials and case reports. We also demonstrate the generalizability of our schema to the computer science and materials science domains. We benchmark state-of-the-art IE systems on CARE, showing that even models such as GPT4 struggle. We release our resources to advance research on extracting and aggregating literature findings.