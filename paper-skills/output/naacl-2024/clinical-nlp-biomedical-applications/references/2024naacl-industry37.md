---
title: "Large Language Models Encode the Practice of Medicine"
source: "https://aclanthology.org/2024.naacl-industry.37/"
categories: ['clinical-nlp-biomedical-applications', 'continual-learning-memory-transfer-llms']
tags: ['clinical-nlp', 'medical-knowledge', 'language-model-probing']
venue: "NAACL 2024"
tldr: "Demonstrates that language models encode medical knowledge and can perform healthcare prediction tasks."
---

# Large Language Models Encode the Practice of Medicine

**Source**: [https://aclanthology.org/2024.naacl-industry.37/](https://aclanthology.org/2024.naacl-industry.37/)

**TLDR**: Demonstrates that language models encode medical knowledge and can perform healthcare prediction tasks.

## Abstract

AbstractHealthcare tasks such as predicting clinical outcomes across medical and surgical populations, disease prediction, predicting patient health journeys, are typically approached with supervised learning on task-specific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of billions of administrative claims, which essentially encapsulates the practice of medicine, offering a unique perspective on patient care and treatment patterns. Our model, MediClaimGPT, a 125M parameter Transformer demonstrates strong zero-shot predictive capabilities, accurately forecasting patient health events across four evaluation datasets, with its capabilities further demonstrated in various downstream tasks. A significant application of MediClaimGPT is in generating high-quality, clinically plausible synthetic claims data, enhancing healthcare data utility while preserving patient privacy. This research underscores the potential of language models in handling complex datasets and their strategic application in healthcare and related fields.