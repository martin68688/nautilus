---
title: "GEE! Grammar Error Explanation with Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.49/"
categories: ['llm-education-assessment-augmentation', 'large-language-model-evaluation-augmentation']
tags: ['grammar-error', 'explanation', 'education']
venue: "NAACL 2024"
tldr: "Uses LLMs to generate natural language explanations for grammar errors, aiding language learning."
---

# GEE! Grammar Error Explanation with Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.49/](https://aclanthology.org/2024.findings-naacl.49/)

**TLDR**: Uses LLMs to generate natural language explanations for grammar errors, aiding language learning.

## Abstract

AbstractExisting grammatical error correction tools do not provide natural language explanations of the errors that they correct in user-written text. However, such explanations are essential for helping users learn the language by gaining a deeper understanding of its grammatical rules (DeKeyser, 2003; Ellis et al., 2006).To address this gap, we propose the task of grammar error explanation, where a system needs to provide one-sentence explanations for each grammatical error in a pair of erroneous and corrected sentences. The task is not easily solved by prompting LLMs: we find that, using one-shot prompting, GPT-4 only explains 40.6% of the errors and does not even attempt to explain 39.8% of the errors.Since LLMs struggle to identify grammar errors, we develop a two-step pipeline that leverages fine-tuned and prompted large language models to perform structured atomic token edit extraction, followed by prompting GPT-4 to explain each edit. We evaluate our pipeline on German, Chinese, and English grammar error correction data. Our atomic edit extraction achieves an F1 of 0.93 on German, 0.91 on Chinese, and 0.891 on English. Human evaluation of generated explanations reveals that 93.9% of German errors, 96.4% of Chinese errors, and 92.20% of English errors are correctly detected and explained. To encourage further research, we open-source our data and code.