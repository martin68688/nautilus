---
title: "Assessing Factual Reliability of Large Language Model Knowledge"
source: "https://aclanthology.org/2024.naacl-long.46/"
categories: ['llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['factuality', 'reliability', 'hallucination', 'evaluation']
venue: "NAACL 2024"
tldr: "Proposes methods to assess the factual reliability and consistency of LLM knowledge beyond accuracy."
---

# Assessing Factual Reliability of Large Language Model Knowledge

**Source**: [https://aclanthology.org/2024.naacl-long.46/](https://aclanthology.org/2024.naacl-long.46/)

**TLDR**: Proposes methods to assess the factual reliability and consistency of LLM knowledge beyond accuracy.

## Abstract

AbstractThe factual knowledge of LLMs is typically evaluated using accuracy, yet this metric does not capture the vulnerability of LLMs to hallucination-inducing factors like prompt and context variability. How do we evaluate the capabilities of LLMs to consistently produce factually correct answers? In this paper, we propose MOdel kNowledge relIabiliTy scORe (MONITOR), a novel metric designed to directly measure LLMs’ factual reliability. MONITOR is designed to compute the distance between the probability distributions of a valid output and its counterparts produced by the same LLM probing the same fact using different styles of prompts and contexts. Experiments on a comprehensive range of 12 LLMs demonstrate the effectiveness of MONITOR in evaluating the factual reliability of LLMs while maintaining a low computational overhead. In addition, we release the FKTC (Factual Knowledge Test Corpus) to foster research along this line https://github.com/Vicky-Wil/MONITOR.