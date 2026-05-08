---
title: "When Life Gives You Lemons, Make Cherryade: Converting Feedback from Bad Responses into Good Labels"
source: "https://aclanthology.org/2024.naacl-long.169/"
categories: ['efficient-large-model-training-optimization', 'llm-backdoor-attacks-defense']
tags: ['dialogue', 'human-feedback', 'continuous-improvement', 'labeling']
venue: "NAACL 2024"
tldr: "A framework (Juicer) converts implicit human feedback on bad chatbot responses into training labels for improvement."
---

# When Life Gives You Lemons, Make Cherryade: Converting Feedback from Bad Responses into Good Labels

**Source**: [https://aclanthology.org/2024.naacl-long.169/](https://aclanthology.org/2024.naacl-long.169/)

**TLDR**: A framework (Juicer) converts implicit human feedback on bad chatbot responses into training labels for improvement.

## Abstract

AbstractDeployed dialogue agents have the potential to integrate human feedback to continuously improve themselves. However, humans may not always provide explicit signals when the chatbot makes mistakes during interactions. In this work, we propose Juicer, a framework to make use of both binary and free-form textual human feedback. It works by: (i) extending sparse binary feedback by training a satisfaction classifier to label the unlabeled data; and (ii) training a reply corrector to map the bad replies to good ones. We find that augmenting training with model-corrected replies improves the final dialogue model, and we can further improve performance by using both positive and negative replies through the recently proposed Director model.