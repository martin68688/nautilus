---
title: "Uncertainty Quantification for In-Context Learning of Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.184/"
categories: ['llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['in-context-learning', 'uncertainty', 'calibration']
venue: "NAACL 2024"
tldr: "Proposes methods for uncertainty quantification in the in-context learning of large language models to improve trustworthiness."
---

# Uncertainty Quantification for In-Context Learning of Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.184/](https://aclanthology.org/2024.naacl-long.184/)

**TLDR**: Proposes methods for uncertainty quantification in the in-context learning of large language models to improve trustworthiness.

## Abstract

AbstractIn-context learning has emerged as a groundbreaking ability of Large Language Models (LLMs) and revolutionized various fields by providing a few task-relevant demonstrations in the prompt. However, trustworthy issues with LLM’s response, such as hallucination, have also been actively discussed. Existing works have been devoted to quantifying the uncertainty in LLM’s response, but they often overlook the complex nature of LLMs and the uniqueness of in-context learning. In this work, we delve into the predictive uncertainty of LLMs associated with in-context learning, highlighting that such uncertainties may stem from both the provided demonstrations (aleatoric uncertainty) and ambiguities tied to the model’s configurations (epistemic uncertainty). We propose a novel formulation and corresponding estimation method to quantify both types of uncertainties. The proposed method offers an unsupervised way to understand the prediction of in-context learning in a plug-and-play fashion. Extensive experiments are conducted to demonstrate the effectiveness of the decomposition. The code and data are available at: https://github.com/lingchen0331/UQ_ICL.