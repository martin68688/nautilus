---
title: "Leveraging LLMs for Dialogue Quality Measurement"
source: "https://aclanthology.org/2024.naacl-industry.30/"
categories: ['llm-evaluation-summarization-argument-extraction', 'legal-ai-nlp-applications']
tags: ['dialogue', 'evaluation', 'llm', 'quality-measurement']
venue: "NAACL 2024"
tldr: "Exploring the use of large language models for measuring dialogue quality in task-oriented systems."
---

# Leveraging LLMs for Dialogue Quality Measurement

**Source**: [https://aclanthology.org/2024.naacl-industry.30/](https://aclanthology.org/2024.naacl-industry.30/)

**TLDR**: Exploring the use of large language models for measuring dialogue quality in task-oriented systems.

## Abstract

AbstractIn task-oriented conversational AI evaluation, unsupervised methods poorly correlate with human judgments, and supervised approaches lack generalization. Recent advances in large language models (LLMs) show robust zero- and few-shot capabilities across NLP tasks. Our paper explores using LLMs for automated dialogue quality evaluation, experimenting with various configurations on public and proprietary datasets. Manipulating factors such as model size, in-context examples, and selection techniques, we examine “chain-of-thought” (CoT) reasoning and label extraction procedures. Our results show that (1) larger models yield more accurate dialogue labels; (2) algorithmic selection of in-context examples outperforms random selection,; (3) CoT reasoning where an LLM is asked to provide justifications before outputting final labels improves performance; and (4) fine-tuned LLMs outperform out-of-the-box ones. In addition, we find that suitably tuned LLMs exhibit high accuracy in dialogue evaluation compared to human judgments.