---
title: "GRASP: A Disagreement Analysis Framework to Assess Group Associations in Perspectives"
source: "https://aclanthology.org/2024.naacl-long.190/"
categories: ['llm-backdoor-attacks-defense', 'human-llm-opinion-dynamics-moderation']
tags: ['annotation', 'subjectivity', 'perspective', 'disagreement']
venue: "NAACL 2024"
tldr: "GRASP is a framework for analyzing disagreement in subjective human annotations to assess group associations in perspectives."
---

# GRASP: A Disagreement Analysis Framework to Assess Group Associations in Perspectives

**Source**: [https://aclanthology.org/2024.naacl-long.190/](https://aclanthology.org/2024.naacl-long.190/)

**TLDR**: GRASP is a framework for analyzing disagreement in subjective human annotations to assess group associations in perspectives.

## Abstract

AbstractHuman annotation plays a core role in machine learning — annotations for supervised models, safety guardrails for generative models, and human feedback for reinforcement learning, to cite a few avenues. However, the fact that many of these human annotations are inherently subjective is often overlooked. Recent work has demonstrated that ignoring rater subjectivity (typically resulting in rater disagreement) is problematic within specific tasks and for specific subgroups. Generalizable methods to harness rater disagreement and thus understand the socio-cultural leanings of subjective tasks remain elusive. In this paper, we propose GRASP, a comprehensive disagreement analysis framework to measure group association in perspectives among different rater subgroups, and demonstrate its utility in assessing the extent of systematic disagreements in two datasets: (1) safety annotations of human-chatbot conversations, and (2) offensiveness annotations of social media posts, both annotated by diverse rater pools across different socio-demographic axes. Our framework (based on disagreement metrics) reveals specific rater groups that have significantly different perspectives than others on certain tasks, and helps identify demographic axes that are crucial to consider in specific task contexts.