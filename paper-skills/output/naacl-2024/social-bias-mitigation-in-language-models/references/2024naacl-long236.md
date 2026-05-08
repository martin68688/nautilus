---
title: "Exploring Cross-Cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis"
source: "https://aclanthology.org/2024.naacl-long.236/"
categories: ['llm-backdoor-attacks-defense', 'social-bias-mitigation-in-language-models', 'social-media-linguistic-variation']
tags: ['hate-speech', 'cross-cultural', 'dataset', 'annotation-bias']
venue: "NAACL 2024"
tldr: "Introduces CREHate, a cross-cultural English hate speech dataset, and explores annotation differences."
---

# Exploring Cross-Cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis

**Source**: [https://aclanthology.org/2024.naacl-long.236/](https://aclanthology.org/2024.naacl-long.236/)

**TLDR**: Introduces CREHate, a cross-cultural English hate speech dataset, and explores annotation differences.

## Abstract

AbstractMost hate speech datasets neglect the cultural diversity within a single language, resulting in a critical shortcoming in hate speech detection. To address this, we introduce CREHate, a CRoss-cultural English Hate speech dataset. To construct CREHate, we follow a two-step procedure: 1) cultural post collection and 2) cross-cultural annotation. We sample posts from the SBIC dataset, which predominantly represents North America, and collect posts from four geographically diverse English-speaking countries (Australia, United Kingdom, Singapore, and South Africa) using culturally hateful keywords we retrieve from our survey. Annotations are collected from the four countries plus the United States to establish representative labels for each country. Our analysis highlights statistically significant disparities across countries in hate speech annotations. Only 56.2% of the posts in CREHate achieve consensus among all countries, with the highest pairwise label difference rate of 26%. Qualitative analysis shows that label disagreement occurs mostly due to different interpretations of sarcasm and the personal bias of annotators on divisive topics. Lastly, we evaluate large language models (LLMs) under a zero-shot setting and show that current LLMs tend to show higher accuracies on Anglosphere country labels in CREHate.Our dataset and codes are available at: https://github.com/nlee0212/CREHate