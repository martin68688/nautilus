---
title: "Improving Pre-trained Language Model Sensitivity via Mask Specific losses: A case study on Biomedical NER"
source: "https://aclanthology.org/2024.naacl-long.280/"
categories: ['efficient-transformer-optimization-techniques', 'clinical-nlp-biomedical-applications']
tags: ['domain-adaptation', 'biomedical-ner', 'mask-specific-loss', 'sensitivity']
venue: "NAACL 2024"
tldr: "This paper proposes improving pre-trained language model sensitivity for biomedical NER via mask-specific losses during fine-tuning."
---

# Improving Pre-trained Language Model Sensitivity via Mask Specific losses: A case study on Biomedical NER

**Source**: [https://aclanthology.org/2024.naacl-long.280/](https://aclanthology.org/2024.naacl-long.280/)

**TLDR**: This paper proposes improving pre-trained language model sensitivity for biomedical NER via mask-specific losses during fine-tuning.

## Abstract

AbstractAdapting language models (LMs) to novel domains is often achieved through fine-tuning a pre-trained LM (PLM) on domain-specific data. Fine-tuning introduces new knowledge into an LM, enabling it to comprehend and efficiently perform a target domain task. Fine-tuning can however be inadvertently insensitive if it ignores the wide array of disparities (e.g in word meaning) between source and target domains. For instance, words such as chronic and pressure may be treated lightly in social conversations, however, clinically, these words are usually an expression of concern. To address insensitive fine-tuning, we propose Mask Specific Language Modeling (MSLM), an approach that efficiently acquires target domain knowledge by appropriately weighting the importance of domain-specific terms (DS-terms) during fine-tuning. MSLM jointly masks DS-terms and generic words, then learns mask-specific losses by ensuring LMs incur larger penalties for inaccurately predicting DS-terms compared to generic words. Results of our analysis show that MSLM improves LMs sensitivity and detection of DS-terms. We empirically show that an optimal masking rate not only depends on the LM, but also on the dataset and the length of sequences. Our proposed masking strategy outperforms advanced masking strategies such as span- and PMI-based masking.