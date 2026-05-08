---
title: "Group Fairness in Multilingual Speech Recognition Models"
source: "https://aclanthology.org/2024.findings-naacl.143/"
categories: ['speech-language-processing-multilingual', 'multilingual-fairness-speech-recognition']
tags: ['speech-recognition', 'fairness', 'multilingual', 'whisper', 'mms']
venue: "NAACL 2024"
tldr: "Performance disparities of multilingual ASR models are evaluated, finding model size correlates logarithmically with worst-case group error and that fine-tuning can reduce bias."
---

# Group Fairness in Multilingual Speech Recognition Models

**Source**: [https://aclanthology.org/2024.findings-naacl.143/](https://aclanthology.org/2024.findings-naacl.143/)

**TLDR**: Performance disparities of multilingual ASR models are evaluated, finding model size correlates logarithmically with worst-case group error and that fine-tuning can reduce bias.

## Abstract

AbstractWe evaluate the performance disparity of the Whisper and MMS families of ASR models across the VoxPopuli and Common Voice multilingual datasets, with an eye toward intersectionality. Our two most important findings are that model size, surprisingly, correlates logarithmically with worst-case performance disparities, meaning that larger (and better) models are less fair. We also observe the importance of intersectionality. In particular, models often exhibit significant performance disparity across binary gender for adolescents.