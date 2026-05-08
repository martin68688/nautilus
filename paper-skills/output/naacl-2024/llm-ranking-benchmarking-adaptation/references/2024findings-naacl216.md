---
title: "Anti-LM Decoding for Zero-shot In-context Machine Translation"
source: "https://aclanthology.org/2024.findings-naacl.216/"
categories: ['llm-ranking-benchmarking-adaptation', 'zero-shot-few-shot-multimodal-optimization']
tags: ['machine-translation', 'zero-shot', 'contrastive-decoding']
venue: "NAACL 2024"
tldr: "Proposes anti-LM decoding, a contrastive decoding approach, to improve zero-shot in-context machine translation by reducing bias."
---

# Anti-LM Decoding for Zero-shot In-context Machine Translation

**Source**: [https://aclanthology.org/2024.findings-naacl.216/](https://aclanthology.org/2024.findings-naacl.216/)

**TLDR**: Proposes anti-LM decoding, a contrastive decoding approach, to improve zero-shot in-context machine translation by reducing bias.

## Abstract

AbstractZero-shot In-context learning is the phenomenon where models can perform a task given only the instructions. However, pre-trained large language models are known to be poorly calibrated for zero-shot tasks. One of the most effective approaches to handling this bias is to adopt a contrastive decoding objective, which accounts for the prior probability of generating the next token by conditioning on a context. This work introduces an Anti-Language Model objective with a decay factor designed to address the weaknesses of In-context Machine Translation. We conduct our experiments across 3 model types and sizes, 3 language directions, and for both greedy decoding and beam search. The proposed method outperforms other state-of-the-art decoding objectives, with up to 20 BLEU point improvement from the default objective in some settings.