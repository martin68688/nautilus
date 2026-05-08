---
title: "Massive End-to-end Speech Recognition Models with Time Reduction"
source: "https://aclanthology.org/2024.naacl-long.344/"
categories: ['llm-evaluation-summarization-argument-extraction', 'speech-language-processing-multilingual']
tags: ['speech-recognition', 'asr', 'efficiency', 'time-reduction']
venue: "NAACL 2024"
tldr: "Massive end-to-end ASR models with funnel pooling for time reduction, improving efficiency while maintaining accuracy."
---

# Massive End-to-end Speech Recognition Models with Time Reduction

**Source**: [https://aclanthology.org/2024.naacl-long.344/](https://aclanthology.org/2024.naacl-long.344/)

**TLDR**: Massive end-to-end ASR models with funnel pooling for time reduction, improving efficiency while maintaining accuracy.

## Abstract

AbstractWe investigate massive end-to-end automatic speech recognition (ASR) models with efficiency improvements achieved by time reduction. The encoders of our models use the neural architecture of Google’s universal speech model (USM), with additional funnel pooling layers to significantly reduce the frame rate and speed up training and inference. We also explore a few practical methods to mitigate potential accuracy loss due to time reduction, while enjoying most efficiency gain. Our methods are demonstrated to work with both Connectionist Temporal Classification (CTC) and RNN-Transducer (RNN-T), with up to 2B model parameters, and over two domains. For a large-scale voice search recognition task, we perform extensive studies on vocabulary size, time reduction strategy, and its generalization performance on long-form test sets, and show that a 900M RNN-T is very tolerant to severe time reduction, with as low encoder output frame rate as 640ms. We also provide ablation studies on the Librispeech benchmark for important training hyperparameters and architecture designs, in training 600M RNN-T models at the frame rate of 160ms.