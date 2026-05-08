---
title: "LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback"
source: "https://aclanthology.org/2024.findings-naacl.92/"
categories: ['llm-backdoor-attacks-defense', 'large-language-model-evaluation-augmentation']
tags: ['llm-refinement', 'feedback', 'optimization']
venue: "NAACL 2024"
tldr: "Proposes LLMRefine, an inference-time method to refine LLM outputs using fine-grained actionable feedback."
---

# LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback

**Source**: [https://aclanthology.org/2024.findings-naacl.92/](https://aclanthology.org/2024.findings-naacl.92/)

**TLDR**: Proposes LLMRefine, an inference-time method to refine LLM outputs using fine-grained actionable feedback.

## Abstract

AbstractRecent large language models (LLM) areleveraging human feedback to improve theirgeneration quality. However, human feedbackis costly to obtain, especially during inference.In this work, we propose LLMRefine, aninference time optimization method to refineLLM’s output. The core idea is to usea learned fine-grained feedback model topinpoint defects and guide LLM to refinethem iteratively. Using original LLM as aproposal of edits, LLMRefine searches fordefect-less text via simulated annealing, tradingoff the exploration and exploitation. Weconduct experiments on three text generationtasks, including machine translation, long-form question answering (QA), and topicalsummarization. LLMRefine consistentlyoutperforms all baseline approaches, achievingimprovements up to 1.7 MetricX points ontranslation tasks, 8.1 ROUGE-L on ASQA, 2.2ROUGE-L on topical summarization.