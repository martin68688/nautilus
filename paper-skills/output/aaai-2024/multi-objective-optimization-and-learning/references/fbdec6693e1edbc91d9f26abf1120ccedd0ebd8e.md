---
title: "MVReward: Better Aligning and Evaluating Multi-View Diffusion Models with Human Preferences"
source: "https://www.semanticscholar.org/paper/fbdec6693e1edbc91d9f26abf1120ccedd0ebd8e"
categories: ['multi-objective-optimization-and-learning', 'audio-multimodal-generation-and-modeling']
tags: ['3d-generation', 'multi-view-diffusion', 'human-preferences', 'evaluation']
venue: "AAAI 2024"
tldr: "A method to better align and evaluate multi-view 3D diffusion models with human preferences through a learned reward model."
---

# MVReward: Better Aligning and Evaluating Multi-View Diffusion Models with Human Preferences

**Source**: [https://www.semanticscholar.org/paper/fbdec6693e1edbc91d9f26abf1120ccedd0ebd8e](https://www.semanticscholar.org/paper/fbdec6693e1edbc91d9f26abf1120ccedd0ebd8e)

**TLDR**: A method to better align and evaluate multi-view 3D diffusion models with human preferences through a learned reward model.

## Abstract

Recent years have witnessed remarkable progress in 3D content generation. However, corresponding evaluation methods struggle to keep pace. Automatic approaches have proven challenging to align with human preferences, and the mixed comparison of text- and image-driven methods often leads to unfair evaluations. In this paper, we present a comprehensive framework to better align and evaluate multi-view diffusion models with human preferences. To begin with, we first collect and filter a standardized image prompt set from DALL·E and Objaverse, which we then use to generate multi-view assets with several multi-view diffusion models. Through a systematic ranking pipeline on these assets, we obtain a human annotation dataset with 16k expert pairwise comparisons and train a reward model, coined MVReward, to effectively encode human preferences. With MVReward, image-driven 3D methods can be evaluated against each other in a more fair and transparent manner. Building on this, we further propose Multi-View Preference Learning (MVP), a plug-and-play multi-view diffusion tuning strategy. Extensive experiments demonstrate that MVReward can serve as a reliable metric and MVP consistently enhances the alignment of multi-view diffusion models with human preferences.