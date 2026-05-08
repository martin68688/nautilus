---
title: "SELF-EXPERTISE: Knowledge-based Instruction Dataset Augmentation for a Legal Expert Language Model"
source: "https://aclanthology.org/2024.findings-naacl.69/"
categories: ['legal-ai-nlp-applications']
tags: ['legal-ai', 'instruction-tuning', 'dataset-augmentation', 'hallucination-reduction']
venue: "NAACL 2024"
tldr: "This paper proposes a knowledge-based instruction dataset augmentation method to reduce hallucinations for a legal expert language model."
---

# SELF-EXPERTISE: Knowledge-based Instruction Dataset Augmentation for a Legal Expert Language Model

**Source**: [https://aclanthology.org/2024.findings-naacl.69/](https://aclanthology.org/2024.findings-naacl.69/)

**TLDR**: This paper proposes a knowledge-based instruction dataset augmentation method to reduce hallucinations for a legal expert language model.

## Abstract

AbstractThe advent of instruction-tuned large language models (LLMs) has significantly advanced the field of automatic instruction dataset augmentation. However, the method of generating instructions and outputs from inherent knowledge of LLM can unintentionally produce hallucinations — instances of generating factually incorrect or misleading information. To overcome this, we propose SELF-EXPERTISE, automatically generating instruction dataset in the legal domain from a seed dataset. SELF-EXPERTISE extracts knowledge from the outputs of the seed dataset, and generates new instructions, inputs, and outputs. In this way, the proposed method reduces hallucination in automatic instruction augmentation. We trained an SELF-EXPERTISE augmented instruction dataset on the LLaMA-2 7B model to construct Korean legal specialized model, called LxPERT. LxPERT has demonstrated performance surpassing GPT-3.5-turbo in both in-domain and out-of-domain datasets. The SELF-EXPERTISE augmentation pipeline is not only applicable to the legal field but is also expected to be extendable to various domains, potentially advancing domain-specialized LLMs.