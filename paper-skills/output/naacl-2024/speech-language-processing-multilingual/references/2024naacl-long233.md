---
title: "Automatic Restoration of Diacritics for Speech Data Sets"
source: "https://aclanthology.org/2024.naacl-long.233/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'speech-language-processing-multilingual']
tags: ['speech', 'diacritics', 'domain-adaptation']
venue: "NAACL 2024"
tldr: "Explores improving automatic diacritic restoration for speech transcripts by addressing domain/style shifts in spoken language."
---

# Automatic Restoration of Diacritics for Speech Data Sets

**Source**: [https://aclanthology.org/2024.naacl-long.233/](https://aclanthology.org/2024.naacl-long.233/)

**TLDR**: Explores improving automatic diacritic restoration for speech transcripts by addressing domain/style shifts in spoken language.

## Abstract

AbstractAutomatic text-based diacritic restoration models generally have high diacritic error rates when applied to speech transcripts as a result of domain and style shifts in spoken language. In this work, we explore the possibility of improving the performance of automatic diacritic restoration when applied to speech data by utilizing parallel spoken utterances. In particular, we use the pre-trained Whisper ASR model fine-tuned on relatively small amounts of diacritized Arabic speech data to produce rough diacritized transcripts for the speech utterances, which we then use as an additional input for diacritic restoration models. The proposed framework consistently improves diacritic restoration performance compared to text-only baselines. Our results highlight the inadequacy of current text-based diacritic restoration models for speech data sets and provide a new baseline for speech-based diacritic restoration.