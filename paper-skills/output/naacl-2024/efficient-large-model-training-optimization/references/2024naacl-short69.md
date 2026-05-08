---
title: "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding"
source: "https://aclanthology.org/2024.naacl-short.69/"
categories: ['efficient-large-model-training-optimization', 'large-language-model-evaluation-augmentation', 'llm-alignment-safety-detoxification']
tags: ['decoding', 'hallucination', 'faithfulness']
venue: "NAACL 2024"
tldr: "Introduces context-aware decoding, a contrastive method to reduce hallucinations by amplifying attention to the input context."
---

# Trusting Your Evidence: Hallucinate Less with Context-aware Decoding

**Source**: [https://aclanthology.org/2024.naacl-short.69/](https://aclanthology.org/2024.naacl-short.69/)

**TLDR**: Introduces context-aware decoding, a contrastive method to reduce hallucinations by amplifying attention to the input context.

## Abstract

AbstractLanguage models (LMs) often struggle to pay enough attention to the input context, and generate texts that are unfaithful or contain hallucinations. To mitigate this issue, we present context-aware decoding (CAD), which follows a contrastive output distribution that amplifies the difference between the output probabilities when a model is used with and without context. Our experiments show that CAD, without additional training, significantly improves the faithfulness of different LM families, including OPT, GPT, LLaMA, and FLAN-T5 for summarization tasks (e.g., 14.3% gain for LLaMA in factuality metrics). Furthermore, CAD is particularly effective in overriding a model’s prior knowledge when it contradicts the provided context, leading to substantial improvements in tasks where resolving the knowledge conflict is essential. Our code is publicly released at https://github.com/xhan77/context-aware-decoding.