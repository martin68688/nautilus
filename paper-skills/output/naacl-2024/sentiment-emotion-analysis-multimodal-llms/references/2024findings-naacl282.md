---
title: "Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation"
source: "https://aclanthology.org/2024.findings-naacl.282/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'human-llm-opinion-dynamics-moderation']
tags: ['emotion-recognition', 'contrastive-learning', 'conversation']
venue: "NAACL 2024"
tldr: "Introduces an emotion-anchored contrastive learning framework for emotion recognition in conversation."
---

# Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation

**Source**: [https://aclanthology.org/2024.findings-naacl.282/](https://aclanthology.org/2024.findings-naacl.282/)

**TLDR**: Introduces an emotion-anchored contrastive learning framework for emotion recognition in conversation.

## Abstract

AbstractEmotion Recognition in Conversation (ERC) involves detecting the underlying emotion behind each utterance within a conversation. Effectively generating representations for utterances remains a significant challenge in this task. Recent works propose various models to address this issue, but they still struggle with differentiating similar emotions such as excitement and happiness. To alleviate this problem, We propose an Emotion-Anchored Contrastive Learning (EACL) framework that can generate more distinguishable utterance representations for similar emotions. To achieve this, we utilize label encodings as anchors to guide the learning of utterance representations and design an auxiliary loss to ensure the effective separation of anchors for similar emotions. Moreover, an additional adaptation process is proposed to adapt anchors to serve as effective classifiers to improve classification performance. Across extensive experiments, our proposed EACL achieves state-of-the-art emotion recognition performance and exhibits superior performance on similar emotions. Our code is available at https://github.com/Yu-Fangxu/EACL.