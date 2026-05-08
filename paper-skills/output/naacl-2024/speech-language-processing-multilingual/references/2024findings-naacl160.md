---
title: "Heterogeneity over Homogeneity: Investigating Multilingual Speech Pre-Trained Models for Detecting Audio Deepfake"
source: "https://aclanthology.org/2024.findings-naacl.160/"
categories: ['code-search-clone-detection', 'speech-language-processing-multilingual', 'multilingual-fairness-speech-recognition']
tags: ['audio-deepfake', 'multilingual', 'pre-trained-models', 'detection']
venue: "NAACL 2024"
tldr: ""
---

# Heterogeneity over Homogeneity: Investigating Multilingual Speech Pre-Trained Models for Detecting Audio Deepfake

**Source**: [https://aclanthology.org/2024.findings-naacl.160/](https://aclanthology.org/2024.findings-naacl.160/)

**TLDR**: 

## Abstract

AbstractIn this work, we investigate multilingual speech Pre-Trained models (PTMs) for Audio deepfake detection (ADD). We hypothesize thatmultilingual PTMs trained on large-scale diverse multilingual data gain knowledge about diverse pitches, accents, and tones, during theirpre-training phase and making them more robust to variations. As a result, they will be more effective for detecting audio deepfakes. To validate our hypothesis, we extract representations from state-of-the-art (SOTA) PTMs including monolingual, multilingual as well as PTMs trained for speaker and emotion recognition, and evaluated them on ASVSpoof 2019 (ASV), In-the-Wild (ITW), and DECRO benchmark databases. We show that representations from multilingual PTMs, with simple downstream networks, attain the best performance for ADD compared to other PTM representations, which validates our hypothesis. We also explore the possibility of fusion of selected PTM representations for further improvements in ADD, and we propose a framework, MiO (Merge into One) for this purpose. With MiO, we achieve SOTA performance on ASV and ITW and comparable performance on DECRO with current SOTA works.