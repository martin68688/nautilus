---
title: "Language Models (Mostly) Do Not Consider Emotion Triggers When Predicting Emotion"
source: "https://aclanthology.org/2024.naacl-short.51/"
categories: ['sentiment-emotion-analysis-multimodal-llms']
tags: ['emotion-detection', 'emotion-triggers', 'model-explainability']
venue: "NAACL 2024"
tldr: "Investigates how well emotion detection models consider human-annotated emotion triggers, finding they mostly do not."
---

# Language Models (Mostly) Do Not Consider Emotion Triggers When Predicting Emotion

**Source**: [https://aclanthology.org/2024.naacl-short.51/](https://aclanthology.org/2024.naacl-short.51/)

**TLDR**: Investigates how well emotion detection models consider human-annotated emotion triggers, finding they mostly do not.

## Abstract

AbstractSituations and events evoke emotions in humans, but to what extent do they inform the prediction of emotion detection models? This work investigates how well human-annotated emotion triggers correlate with features that models deemed salient in their prediction of emotions. First, we introduce a novel dataset EmoTrigger, consisting of 900 social media posts sourced from three different datasets; these were annotated by experts for emotion triggers with high agreement. Using EmoTrigger, we evaluate the ability of large language models (LLMs) to identify emotion triggers, and conduct a comparative analysis of the features considered important for these tasks between LLMs and fine-tuned models. Our analysis reveals that emotion triggers are largely not considered salient features for emotion prediction models, instead there is intricate interplay between various features and the task of emotion detection.