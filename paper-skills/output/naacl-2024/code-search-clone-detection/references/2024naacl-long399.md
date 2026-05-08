---
title: "Towards Reducing Diagnostic Errors with Interpretable Risk Prediction"
source: "https://aclanthology.org/2024.naacl-long.399/"
categories: ['clinical-nlp-biomedical-applications', 'code-search-clone-detection', 'clinical-note-generation-llm-benchmarking']
tags: ['diagnostic', 'interpretable', 'risk-prediction', 'EHR']
venue: "NAACL 2024"
tldr: "Uses LLMs to identify evidence in EHRs for interpretable risk prediction to help reduce diagnostic errors."
---

# Towards Reducing Diagnostic Errors with Interpretable Risk Prediction

**Source**: [https://aclanthology.org/2024.naacl-long.399/](https://aclanthology.org/2024.naacl-long.399/)

**TLDR**: Uses LLMs to identify evidence in EHRs for interpretable risk prediction to help reduce diagnostic errors.

## Abstract

AbstractMany diagnostic errors occur because clinicians cannot easily access relevant information in patient Electronic Health Records (EHRs). In this work we propose a method to use LLMs to identify pieces of evidence in patient EHR data that indicate increased or decreased risk of specific diagnoses; our ultimate aim is to increase access to evidence and reduce diagnostic errors. In particular, we propose a Neural Additive Model to make predictions backed by evidence with individualized risk estimates at time-points where clinicians are still uncertain, aiming to specifically mitigate delays in diagnosis and errors stemming from an incomplete differential. To train such a model, it is necessary to infer temporally fine-grained retrospective labels of eventual “true” diagnoses. We do so with LLMs, to ensure that the input text is from before a confident diagnosis can be made. We use an LLM to retrieve an initial pool of evidence, but then refine this set of evidence according to correlations learned by the model. We conduct an in-depth evaluation of the usefulness of our approach by simulating how it might be used by a clinician to decide between a pre-defined list of differential diagnoses.