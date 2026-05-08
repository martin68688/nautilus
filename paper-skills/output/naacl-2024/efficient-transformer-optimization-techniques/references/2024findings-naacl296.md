---
title: "DecoderLens: Layerwise Interpretation of Encoder-Decoder Transformers"
source: "https://aclanthology.org/2024.findings-naacl.296/"
categories: ['efficient-transformer-optimization-techniques', 'transformer-language-model-probing']
tags: ['transformer-interpretability', 'encoder-decoder', 'model-analysis']
venue: "NAACL 2024"
tldr: "Proposes DecoderLens, a method to analyze encoder-decoder Transformers by reading internal layer representations with the original decoder."
---

# DecoderLens: Layerwise Interpretation of Encoder-Decoder Transformers

**Source**: [https://aclanthology.org/2024.findings-naacl.296/](https://aclanthology.org/2024.findings-naacl.296/)

**TLDR**: Proposes DecoderLens, a method to analyze encoder-decoder Transformers by reading internal layer representations with the original decoder.

## Abstract

AbstractIn recent years, several interpretability methods have been proposed to interpret the inner workings of Transformer models at different levels of precision and complexity.In this work, we propose a simple but effective technique to analyze encoder-decoder Transformers. Our method, which we name DecoderLens, allows the decoder to cross-attend representations of intermediate encoder activations instead of using the default final encoder output.The method thus maps uninterpretable intermediate vector representations to human-interpretable sequences of words or symbols, shedding new light on the information flow in this popular but understudied class of models.We apply DecoderLens to question answering, logical reasoning, speech recognition and machine translation models, finding that simpler subtasks are solved with high precision by low and intermediate encoder layers.