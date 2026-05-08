---
title: "Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions"
source: "https://aclanthology.org/2024.findings-naacl.130/"
categories: ['large-language-model-evaluation-augmentation', 'language-model-evaluation-benchmarks']
tags: ['evaluation-bias', 'prompt-sensitivity', 'multiple-choice']
venue: "NAACL 2024"
tldr: "It investigates LLM sensitivity to the order of options in multiple-choice questions, highlighting a challenge for fair model assessment."
---

# Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions

**Source**: [https://aclanthology.org/2024.findings-naacl.130/](https://aclanthology.org/2024.findings-naacl.130/)

**TLDR**: It investigates LLM sensitivity to the order of options in multiple-choice questions, highlighting a challenge for fair model assessment.

## Abstract

AbstractLarge Language Models (LLMs) have demonstrated remarkable capabilities in various NLP tasks. However, previous works have shown these models are sensitive towards prompt wording, and few-shot demonstrations and their order, posing challenges to fair assessment of these models. As these models become more powerful, it becomes imperative to understand and address these limitations. In this paper, we focus on LLMs robustness on the task of multiple-choice questions—commonly adopted task to study reasoning and fact-retrieving capability of LLMs. Investigating the sensitivity of LLMs towards the order of options in multiple-choice questions, we demonstrate a considerable performance gap of approximately 13% to 85% in LLMs on different benchmarks, when answer options are reordered, even when using demonstrations in a few-shot setting. Through a detailed analysis, we conjecture that this sensitivity arises when LLMs are uncertain about the prediction between the top-2/3 choices, and specific options placements may favor certain prediction between those top choices depending on the question caused by positional bias. We also identify patterns in top-2 choices that amplify or mitigate the model’s bias toward option placement. We found that for amplifying bias, the optimal strategy involves positioning the top two choices as the first and last options. Conversely, to mitigate bias, we recommend placing these choices among the adjacent options. To validate our conjecture, we conduct various experiments and adopt two approaches to calibrate LLMs’ predictions, leading to up to 8 percentage points improvement across different models and benchmarks.