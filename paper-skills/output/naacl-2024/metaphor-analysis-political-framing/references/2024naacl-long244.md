---
title: "ODD: A Benchmark Dataset for the Natural Language Processing Based Opioid Related Aberrant Behavior Detection"
source: "https://aclanthology.org/2024.naacl-long.244/"
categories: ['clinical-nlp-biomedical-applications', 'metaphor-analysis-political-framing']
tags: ['clinical-nlp', 'opioid', 'behavior-detection']
venue: "NAACL 2024"
tldr: "Introduces a benchmark dataset for detecting opioid-related aberrant behaviors from patient clinical notes."
---

# ODD: A Benchmark Dataset for the Natural Language Processing Based Opioid Related Aberrant Behavior Detection

**Source**: [https://aclanthology.org/2024.naacl-long.244/](https://aclanthology.org/2024.naacl-long.244/)

**TLDR**: Introduces a benchmark dataset for detecting opioid-related aberrant behaviors from patient clinical notes.

## Abstract

AbstractOpioid related aberrant behaviors (ORABs) present novel risk factors for opioid overdose. This paper introduces a novel biomedical natural language processing benchmark dataset named ODD, for ORAB Detection Dataset. ODD is an expert-annotated dataset designed to identify ORABs from patients’ EHR notes and classify them into nine categories; 1) Confirmed Aberrant Behavior, 2) Suggested Aberrant Behavior, 3) Opioids, 4) Indication, 5) Diagnosed opioid dependency, 6) Benzodiazepines, 7) Medication Changes, 8) Central Nervous System-related, and 9) Social Determinants of Health. We explored two state-of-the-art natural language processing models (fine-tuning and prompt-tuning approaches) to identify ORAB. Experimental results show that the prompt-tuning models outperformed the fine-tuning models in most categories and the gains were especially higher among uncommon categories (Suggested Aberrant Behavior, Confirmed Aberrant Behaviors, Diagnosed Opioid Dependence, and Medication Change). Although the best model achieved the highest 88.17% on macro average area under precision recall curve, uncommon classes still have a large room for performance improvement. ODD is publicly available.