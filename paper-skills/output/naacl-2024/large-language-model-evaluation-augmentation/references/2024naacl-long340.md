---
title: "A Study on the Calibration of In-context Learning"
source: "https://aclanthology.org/2024.naacl-long.340/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['knowledge-graphs', 'hallucination', 'survey']
venue: "NAACL 2024"
tldr: "A survey on using knowledge graphs to reduce hallucinations in LLMs by augmenting them with external knowledge."
---

# A Study on the Calibration of In-context Learning

**Source**: [https://aclanthology.org/2024.naacl-long.340/](https://aclanthology.org/2024.naacl-long.340/)

**TLDR**: A survey on using knowledge graphs to reduce hallucinations in LLMs by augmenting them with external knowledge.

## Abstract

AbstractAccurate uncertainty quantification is crucial for the safe deployment of machine learning models, and prior research has demonstrated improvements in the calibration of modern language models (LMs). We study in-context learning (ICL), a prevalent method for adapting static LMs through tailored prompts, and examine the balance between performance and calibration across a broad spectrum of natural language understanding and reasoning tasks. Through comprehensive experiments, we observe that, with an increasing number of ICL examples, models initially exhibit increased miscalibration before achieving better calibration and miscalibration tends to arise in low-shot settings. Moreover, we find that methods aimed at improving usability, such as fine-tuning and chain-of-thought (CoT) prompting, can lead to miscalibration and unreliable natural language explanations. Furthermore, we explore recalibration techniques and find that a scaling-binning calibrator can reduce calibration errors consistently.