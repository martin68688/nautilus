---
title: "EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning"
source: "https://aclanthology.org/2024.naacl-short.35/"
categories: ['zero-shot-few-shot-multimodal-optimization']
tags: ['in-context-learning', 'prompting', 'query-rephrasing', 'optimization']
venue: "NAACL 2024"
tldr: "Improving in-context learning by prompting the model to rephrase its own queries."
---

# EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning

**Source**: [https://aclanthology.org/2024.naacl-short.35/](https://aclanthology.org/2024.naacl-short.35/)

**TLDR**: Improving in-context learning by prompting the model to rephrase its own queries.

## Abstract

AbstractLanguage models are achieving impressive performance on various tasks by aggressively adopting inference-time prompting techniques,such as zero-shot and few-shot prompting. In this work, we introduce EchoPrompt, a simple yet effective approach that prompts the model to rephrase its queries before answering them. EchoPrompt is tailored for four scenarios, including standard and chain-of-thought prompting, in both zero-shot and few-shot settings. Experimental results show that EchoPrompt yields substantial improvementsacross all these settings for four families of causal language models. These improvements are observed across various numerical reasoning (e.g., GSM8K, SVAMP), reading comprehension (e.g., DROP), and logical reasoning (e.g., Coin flipping) tasks. On average, EchoPrompt improves the Zero-shot-CoT performance of code-davinci-002 by 5% in numerical tasks and 13% in reading comprehension tasks. Our empirical results indicate that EchoPrompt is an effective technique that enhances in-context learning performance.