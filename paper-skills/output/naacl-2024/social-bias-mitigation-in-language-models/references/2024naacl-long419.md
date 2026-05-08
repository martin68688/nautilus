---
title: "MisgenderMender: A Community-Informed Approach to Interventions for Misgendering"
source: "https://aclanthology.org/2024.naacl-long.419/"
categories: ['human-llm-opinion-dynamics-moderation', 'social-bias-mitigation-in-language-models']
tags: ['misgendering', 'bias-mitigation', 'community-informed']
venue: "NAACL 2024"
tldr: "Presents a community-informed tool to detect and suggest corrections for misgendering in text."
---

# MisgenderMender: A Community-Informed Approach to Interventions for Misgendering

**Source**: [https://aclanthology.org/2024.naacl-long.419/](https://aclanthology.org/2024.naacl-long.419/)

**TLDR**: Presents a community-informed tool to detect and suggest corrections for misgendering in text.

## Abstract

AbstractContent Warning: This paper contains examples of misgendering and erasure that could be offensive and potentially triggering.Misgendering, the act of incorrectly addressing someone’s gender, inflicts serious harm and is pervasive in everyday technologies, yet there is a notable lack of research to combat it. We are the first to address this lack of research into interventions for misgendering by conducting a survey of gender-diverse individuals in the US to understand perspectives about automated interventions for text-based misgendering. Based on survey insights on the prevalence of misgendering, desired solutions, and associated concerns, we introduce a misgendering interventions task and evaluation dataset, MisgenderMender. We define the task with two sub-tasks: (i) detecting misgendering, followed by (ii) correcting misgendering where misgendering is present, in domains where editing is appropriate. MisgenderMender comprises 3790 instances of social media content and LLM-generations about non-cisgender public figures, annotated for the presence of misgendering, with additional annotations for correcting misgendering in LLM-generated text. Using this dataset, we set initial benchmarks by evaluating existing NLP systems and highlighting challenges for future models to address. We release the full dataset, code, and demo at https://tamannahossainkay.github.io/misgendermender/