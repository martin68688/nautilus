---
title: "A Comprehensive Study of Gender Bias in Chemical Named Entity Recognition Models"
source: "https://aclanthology.org/2024.naacl-long.245/"
categories: ['llm-alignment-safety-detoxification', 'social-bias-mitigation-in-language-models', 'llm-fairness-bias-social-context']
tags: ['gender-bias', 'named-entity-recognition', 'chemical-ner']
venue: "NAACL 2024"
tldr: "Conducts a comprehensive study of gender bias in chemical named entity recognition models."
---

# A Comprehensive Study of Gender Bias in Chemical Named Entity Recognition Models

**Source**: [https://aclanthology.org/2024.naacl-long.245/](https://aclanthology.org/2024.naacl-long.245/)

**TLDR**: Conducts a comprehensive study of gender bias in chemical named entity recognition models.

## Abstract

AbstractChemical named entity recognition (NER) models are used in many downstream tasks, from adverse drug reaction identification to pharmacoepidemiology. However, it is unknown whether these models work the same for everyone. Performance disparities can potentially cause harm rather than the intended good. This paper assesses gender-related performance disparities in chemical NER systems. We develop a framework for measuring gender bias in chemical NER models using synthetic data and a newly annotated corpus of over 92,405 words with self-identified gender information from Reddit. Our evaluation of multiple biomedical NER models reveals evident biases. For instance, synthetic data suggests that female names are frequently misclassified as chemicals, especially when it comes to brand name mentions. Additionally, we observe performance disparities between female- and male-associated data in both datasets. Many systems fail to detect contraceptives such as birth control. Our findings emphasize the biases in chemical NER models, urging practitioners to account for these biases in downstream applications.