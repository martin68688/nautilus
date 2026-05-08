---
title: "Unlocking Parameter-Efficient Fine-Tuning for Low-Resource Language Translation"
source: "https://aclanthology.org/2024.findings-naacl.263/"
categories: ['speech-language-processing-multilingual', 'human-llm-opinion-dynamics-moderation']
tags: ['low-resource', 'translation', 'parameter-efficient-finetuning']
venue: "NAACL 2024"
tldr: "Investigates parameter-efficient fine-tuning methods for low-resource language translation."
---

# Unlocking Parameter-Efficient Fine-Tuning for Low-Resource Language Translation

**Source**: [https://aclanthology.org/2024.findings-naacl.263/](https://aclanthology.org/2024.findings-naacl.263/)

**TLDR**: Investigates parameter-efficient fine-tuning methods for low-resource language translation.

## Abstract

AbstractParameter-efficient fine-tuning (PEFT) methods are increasingly vital in adapting large-scale pre-trained language models for diverse tasks, offering a balance between adaptability and computational efficiency. They are important in Low-Resource Language (LRL) Neural Machine Translation (NMT) to enhance translation accuracy with minimal resources. However, their practical effectiveness varies significantly across different languages. We conducted comprehensive empirical experiments with varying LRL domains and sizes to evaluate the performance of 8 PEFT methods with in total of 15 architectures using the SacreBLEU score. We showed that 6 PEFT architectures outperform the baseline for both in-domain and out-domain tests and the Houlsby+Inversion adapter has the best performance overall, proving the effectiveness of PEFT methods.