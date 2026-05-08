---
title: "DLM: A Decoupled Learning Model for Long-tailed Polyphone Disambiguation in Mandarin"
source: "https://aclanthology.org/2024.naacl-long.294/"
categories: ['legal-ai-nlp-applications', 'speech-language-processing-multilingual']
tags: ['polyphone-disambiguation', 'long-tailed', 'mandarin', 'decoupled-learning']
venue: "NAACL 2024"
tldr: "DLM is a decoupled learning model for long-tailed polyphone disambiguation in Mandarin G2P conversion."
---

# DLM: A Decoupled Learning Model for Long-tailed Polyphone Disambiguation in Mandarin

**Source**: [https://aclanthology.org/2024.naacl-long.294/](https://aclanthology.org/2024.naacl-long.294/)

**TLDR**: DLM is a decoupled learning model for long-tailed polyphone disambiguation in Mandarin G2P conversion.

## Abstract

AbstractGrapheme-to-phoneme conversion (G2P) is a critical component of the text-to-speech system (TTS), where polyphone disambiguation is the most crucial task. However, polyphone disambiguation datasets often suffer from the long-tail problem, and context learning for polyphonic characters commonly stems from a single dimension. In this paper, we propose a novel model DLM: a Decoupled Learning Model for long-tailed polyphone disambiguation in Mandarin. Firstly, DLM decouples representation and classification learnings. It can apply different data samplers for each stage to obtain an optimal training data distribution. This can mitigate the long-tail problem. Secondly, two improved attention mechanisms and a gradual conversion strategy are integrated into the DLM, which achieve transition learning of context from local to global. Finally, to evaluate the effectiveness of DLM, we construct a balanced polyphone disambiguation corpus via in-context learning. Experiments on the benchmark CPP dataset demonstrate that DLM achieves a boosted accuracy of 99.07%. Moreover, DLM improves the disambiguation performance of long-tailed polyphonic characters. For many long-tailed characters, DLM even achieves an accuracy of 100%.