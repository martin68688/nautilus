---
title: "Deferred NAM: Low-latency Top-K Context Injection via Deferred Context Encoding for Non-Streaming ASR"
source: "https://aclanthology.org/2024.naacl-industry.26/"
categories: ['speech-language-processing-multilingual']
tags: ['speech-recognition', 'contextual-biasing', 'low-latency']
venue: "NAACL 2024"
tldr: "Proposes Deferred NAM, a method for low-latency top-K context injection in non-streaming ASR via deferred context encoding."
---

# Deferred NAM: Low-latency Top-K Context Injection via Deferred Context Encoding for Non-Streaming ASR

**Source**: [https://aclanthology.org/2024.naacl-industry.26/](https://aclanthology.org/2024.naacl-industry.26/)

**TLDR**: Proposes Deferred NAM, a method for low-latency top-K context injection in non-streaming ASR via deferred context encoding.

## Abstract

AbstractContextual biasing enables speech recognizers to transcribe important phrases in the speaker’s context, such as contact names, even if they are rare in, or absent from, the training data. Attention-based biasing is a leading approach which allows for full end-to-end cotraining of the recognizer and biasing system and requires no separate inference-time components. Such biasers typically consist of a context encoder; followed by a context filter which narrows down the context to apply, improving per-step inference time; and, finally, context application via cross attention. Though much work has gone into optimizing per-frame performance, the context encoder is at least as important: recognition cannot begin before context encoding ends. Here, we show the lightweight phrase selection pass can be moved before context encoding, resulting in a speedup of up to 16.1 times and enabling biasing to scale to 20K phrases with a maximum pre-decoding delay under 33ms. With the addition of phrase- and wordpiece-level cross-entropy losses, our technique also achieves up to a 37.5% relative WER reduction over the baseline without the losses and lightweight phrase selection pass.