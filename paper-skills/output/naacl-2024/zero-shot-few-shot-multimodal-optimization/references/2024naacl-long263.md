---
title: "Fake Alignment: Are LLMs Really Aligned Well?"
source: "https://aclanthology.org/2024.naacl-long.263/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-alignment-safety-detoxification']
tags: ['alignment', 'safety', 'evaluation', 'llm']
venue: "NAACL 2024"
tldr: "Investigates fake alignment in LLMs, revealing a performance gap between multiple-choice and open-ended safety evaluations."
---

# Fake Alignment: Are LLMs Really Aligned Well?

**Source**: [https://aclanthology.org/2024.naacl-long.263/](https://aclanthology.org/2024.naacl-long.263/)

**TLDR**: Investigates fake alignment in LLMs, revealing a performance gap between multiple-choice and open-ended safety evaluations.

## Abstract

AbstractThe growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety. This study investigates an under-explored issue about the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, LLM only remembers the answer style for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. We introduce a Fake alIgNment Evaluation (FINE) framework and two novel metrics——Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimation. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Subsequently, we found that multiple-choice format data can also be used as high-quality contrast distillation-based fine-tuning data, which can strongly improve the alignment consistency of LLMs with minimal fine-tuning overhead. For data and code, see https://github.com/AIFlames/Fake-Alignment.