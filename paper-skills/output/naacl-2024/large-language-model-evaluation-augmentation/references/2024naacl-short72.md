---
title: "Improved Text Emotion Prediction Using Combined Valence and Arousal Ordinal Classification"
source: "https://aclanthology.org/2024.naacl-short.72/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'large-language-model-evaluation-augmentation']
tags: ['emotion-detection', 'valence-arousal', 'ordinal-classification']
venue: "NAACL 2024"
tldr: "Improves text emotion prediction by using a combined ordinal classification approach for valence and arousal dimensions."
---

# Improved Text Emotion Prediction Using Combined Valence and Arousal Ordinal Classification

**Source**: [https://aclanthology.org/2024.naacl-short.72/](https://aclanthology.org/2024.naacl-short.72/)

**TLDR**: Improves text emotion prediction by using a combined ordinal classification approach for valence and arousal dimensions.

## Abstract

AbstractEmotion detection in textual data has received growing interest in recent years, as it is pivotal for developing empathetic human-computer interaction systems.This paper introduces a method for categorizing emotions from text, which acknowledges and differentiates between the diversified similarities and distinctions of various emotions.Initially, we establish a baseline by training a transformer-based model for standard emotion classification, achieving state-of-the-art performance. We argue that not all misclassifications are of the same importance, as there are perceptual similarities among emotional classes.We thus redefine the emotion labeling problem by shifting it from a traditional classification model to an ordinal classification one, where discrete emotions are arranged in a sequential order according to their valence levels.Finally, we propose a method that performs ordinal classification in the two-dimensional emotion space, considering both valence and arousal scales.The results show that our approach not only preserves high accuracy in emotion prediction but also significantly reduces the magnitude of errors in cases of misclassification.