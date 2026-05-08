---
title: "Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model"
source: "https://aclanthology.org/2024.naacl-long.451/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['reward-model', 'human-feedback', 'translation', 'quality-estimation']
venue: "NAACL 2024"
tldr: "Using quality estimation as a reward model to improve machine translation with human feedback."
---

# Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model

**Source**: [https://aclanthology.org/2024.naacl-long.451/](https://aclanthology.org/2024.naacl-long.451/)

**TLDR**: Using quality estimation as a reward model to improve machine translation with human feedback.

## Abstract

AbstractInsufficient modeling of human preferences within the reward model is a major obstacle for leveraging human feedback to improve translation quality. Fortunately, quality estimation (QE), which predicts the quality of a given translation without reference, has achieved impressive alignment with human evaluations in the last two years. In this work, we investigate the potential of employing the QE model as the reward model to predict human preferences for feedback training. We first identify the overoptimization problem during QE-based feedback training, manifested as an increase in reward while translation quality declines. We examine the problem and argue that the vulnerability of the QE model might lead to high rewards for incorrect translations, resulting in overoptimization and error propagation. To address the problem, we adopt a simple yet effective method that uses heuristic rules to detect the incorrect translations and assigns a penalty term to the reward scores of them. Experimental results show that the proposed QE-based feedback training achieves consistent and significant improvements across various settings, further verified through human preference studies. Our subsequent analysis demonstrates the high data efficiency of the proposed QE-based feedback training: it outperforms systems using larger parallel corpora by a small amount of monolingual data. Our code is available at: https://github.com/zwhe99/FeedbackMT