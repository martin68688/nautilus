---
title: "Improving Factuality in Clinical Abstractive Multi-Document Summarization by Guided Continued Pre-training"
source: "https://aclanthology.org/2024.naacl-short.66/"
categories: ['clinical-nlp-biomedical-applications', 'zero-shot-few-shot-multimodal-optimization']
tags: ['clinical-nlp', 'summarization', 'factuality', 'continued-pretraining']
venue: "NAACL 2024"
tldr: "Uses guided continued pre-training to improve factual accuracy in clinical multi-document summarization."
---

# Improving Factuality in Clinical Abstractive Multi-Document Summarization by Guided Continued Pre-training

**Source**: [https://aclanthology.org/2024.naacl-short.66/](https://aclanthology.org/2024.naacl-short.66/)

**TLDR**: Uses guided continued pre-training to improve factual accuracy in clinical multi-document summarization.

## Abstract

AbstractFactual accuracy is an important property of neural abstractive summarization models, especially in fact-critical domains such as the clinical literature. In this work, we introduce a guided continued pre-training stage for encoder-decoder models that improves their understanding of the factual attributes of documents, which is followed by supervised fine-tuning on summarization. Our approach extends the pre-training recipe of BART to incorporate 3 additional objectives based on PICO spans, which capture the population, intervention, comparison, and outcomes related to a clinical study. Experiments on multi-document summarization in the clinical domain demonstrate that our approach is competitive with prior work, improving the quality and factuality of the summaries and achieving the best-published results in factual accuracy on the MSLR task.