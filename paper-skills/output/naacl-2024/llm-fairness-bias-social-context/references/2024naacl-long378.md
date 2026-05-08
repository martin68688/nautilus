---
title: "Beyond Performance: Quantifying and Mitigating Label Bias in LLMs"
source: "https://aclanthology.org/2024.naacl-long.378/"
categories: ['llm-fairness-bias-social-context']
tags: ['bias', 'label-bias', 'fairness', 'evaluation']
venue: "NAACL 2024"
tldr: "Quantifies and mitigates label bias in LLMs, where models show undesirable preference for certain answer labels."
---

# Beyond Performance: Quantifying and Mitigating Label Bias in LLMs

**Source**: [https://aclanthology.org/2024.naacl-long.378/](https://aclanthology.org/2024.naacl-long.378/)

**TLDR**: Quantifies and mitigates label bias in LLMs, where models show undesirable preference for certain answer labels.

## Abstract

AbstractLarge language models (LLMs) have shown remarkable adaptability to diverse tasks, by leveraging context prompts containing instructions, or minimal input-output examples. However, recent work revealed they also exhibit *label bias*—an undesirable preference toward predicting certain answers over others. Still, detecting and measuring this bias reliably and at scale has remained relatively unexplored. In this study, we evaluate different approaches to quantifying label bias in a model’s predictions, conducting a comprehensive investigation across 279 classification tasks and ten LLMs. Our investigation reveals substantial label bias in models both before and after debiasing attempts, as well as highlights the importance of outcomes-based evaluation metrics, which were not previously used in this regard. We further propose a novel label bias calibration method tailored for few-shot prompting, which outperforms recent calibration approaches for both improving performance and mitigating label bias. Our results emphasize that label bias in the predictions of LLMs remains a barrier to their reliability.