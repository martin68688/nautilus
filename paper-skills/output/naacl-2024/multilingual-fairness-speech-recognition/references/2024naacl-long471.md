---
title: "Multilingual Models for ASR in Chibchan Languages"
source: "https://aclanthology.org/2024.naacl-long.471/"
categories: ['speech-language-processing-multilingual', 'multilingual-fairness-speech-recognition']
tags: ['speech-recognition', 'low-resource', 'chibchan']
venue: "NAACL 2024"
tldr: "Fine-tunes ASR models for Bribri and Cabécar and explores cross-lingual transfer within the Chibchan language family."
---

# Multilingual Models for ASR in Chibchan Languages

**Source**: [https://aclanthology.org/2024.naacl-long.471/](https://aclanthology.org/2024.naacl-long.471/)

**TLDR**: Fine-tunes ASR models for Bribri and Cabécar and explores cross-lingual transfer within the Chibchan language family.

## Abstract

AbstractWe present experiments on Automatic Speech Recognition (ASR) for Bribri and Cabécar, two languages from the Chibchan family. We fine-tune four ASR algorithms (Wav2Vec2, Whisper, MMS & WavLM) to create monolingual models, with the Wav2Vec2 model demonstrating the best performance. We then proceed to use Wav2Vec2 for (1) experiments on training joint and transfer learning models for both languages, and (2) an analysis of the errors, with a focus on the transcription of tone. Results show effective transfer learning for both Bribri and Cabécar, but especially for Bribri. A post-processing spell checking step further reduced character and word error rates. As for the errors, tone is where the Bribri models make the most errors, whereas the simpler tonal system of Cabécar is better transcribed by the model. Our work contributes to developing better ASR technology, an important tool that could facilitate transcription, one of the major bottlenecks in language documentation efforts. Our work also assesses how existing pre-trained models and algorithms perform for genuine extremely low resource-languages.