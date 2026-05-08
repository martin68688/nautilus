---
title: "Reinforced Multiple Instance Selection for Speaker Attribute Prediction"
source: "https://aclanthology.org/2024.naacl-long.181/"
categories: ['human-llm-opinion-dynamics-moderation', 'social-media-linguistic-variation']
tags: ['speaker-attribute', 'multiple-instance-learning', 'reinforcement-learning']
venue: "NAACL 2024"
tldr: "A reinforced multiple instance selection method for predicting speaker attributes from utterances."
---

# Reinforced Multiple Instance Selection for Speaker Attribute Prediction

**Source**: [https://aclanthology.org/2024.naacl-long.181/](https://aclanthology.org/2024.naacl-long.181/)

**TLDR**: A reinforced multiple instance selection method for predicting speaker attributes from utterances.

## Abstract

AbstractLanguage usage is related to speaker age, gender, moral concerns, political ideology, and other attributes. Current state-of-the-art methods for predicting these attributes take a speaker’s utterances as input and provide a prediction per speaker attribute. Most of these approaches struggle to handle a large number of utterances per speaker. This difficulty is primarily due to the computational constraints of the models. Additionally, only a subset of speaker utterances may be relevant to specific attributes. In this paper, we formulate speaker attribute prediction as a Multiple Instance Learning (MIL) problem and propose RL-MIL, a novel approach based on Reinforcement Learning (RL) that effectively addresses both of these challenges. Our experiments demonstrate that our RL-based methodology consistently outperforms previous approaches across a range of related tasks: predicting speakers’ psychographics and demographics from social media posts, and political ideologies from transcribed speeches. We create synthetic datasets and investigate the behavior of RL-MIL systematically. Our results show the success of RL-MIL in improving speaker attribute prediction by learning to select relevant speaker utterances.