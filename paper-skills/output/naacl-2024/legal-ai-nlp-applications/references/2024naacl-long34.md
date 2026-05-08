---
title: "PILOT: Legal Case Outcome Prediction with Case Law"
source: "https://aclanthology.org/2024.naacl-long.34/"
categories: ['legal-ai-nlp-applications', 'knowledge-graph-and-information-extraction']
tags: ['legal-ai', 'case-outcome-prediction', 'case-law']
venue: "NAACL 2024"
tldr: "Addresses legal case outcome prediction in case law systems by identifying relevant precedents and modeling legal reasoning."
---

# PILOT: Legal Case Outcome Prediction with Case Law

**Source**: [https://aclanthology.org/2024.naacl-long.34/](https://aclanthology.org/2024.naacl-long.34/)

**TLDR**: Addresses legal case outcome prediction in case law systems by identifying relevant precedents and modeling legal reasoning.

## Abstract

AbstractMachine learning shows promise in predicting the outcome of legal cases, but most research has concentrated on civil law cases rather than case law systems. We identified two unique challenges in making legal case outcome predictions with case law. First, it is crucial to identify relevant precedent cases that serve as fundamental evidence for judges during decision-making. Second, it is necessary to consider the evolution of legal principles over time, as early cases may adhere to different legal contexts. In this paper, we proposed a new framework named PILOT (PredictIng Legal case OuTcome) for case outcome prediction. It comprises two modules for relevant case retrieval and temporal pattern handling, respectively. To benchmark the performance of existing legal case outcome prediction models, we curated a dataset from a large-scale case law database. We demonstrate the importance of accurately identifying precedent cases and mitigating the temporal shift when making predictions for case law, as our method shows a significant improvement over the prior methods that focus on civil law case outcome predictions.