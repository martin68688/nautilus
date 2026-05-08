---
title: "Generalizable Sarcasm Detection is Just Around the Corner, of Course!"
source: "https://aclanthology.org/2024.naacl-long.238/"
categories: ['llm-evaluation-summarization-argument-extraction', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['sarcasm-detection', 'generalization', 'robustness']
venue: "NAACL 2024"
tldr: "The robustness of sarcasm detection models is tested across datasets with varying characteristics, revealing generalization challenges."
---

# Generalizable Sarcasm Detection is Just Around the Corner, of Course!

**Source**: [https://aclanthology.org/2024.naacl-long.238/](https://aclanthology.org/2024.naacl-long.238/)

**TLDR**: The robustness of sarcasm detection models is tested across datasets with varying characteristics, revealing generalization challenges.

## Abstract

AbstractWe tested the robustness of sarcasm detection models by examining their behavior when fine-tuned on four sarcasm datasets containing varying characteristics of sarcasm: label source (authors vs. third-party), domain (social media/online vs. offline conversations/dialogues), style (aggressive vs. humorous mocking). We tested their prediction performance on the same dataset (intra-dataset) and across different datasets (cross-dataset). For intra-dataset predictions, models consistently performed better when fine-tuned with third-party labels rather than with author labels. For cross-dataset predictions, most models failed to generalize well to the other datasets, implying that one type of dataset cannot represent all sorts of sarcasm with different styles and domains. Compared to the existing datasets, models fine-tuned on the new dataset we release in this work showed the highest generalizability to other datasets. With a manual inspection of the datasets and post-hoc analysis, we attributed the difficulty in generalization to the fact that sarcasm actually comes in different domains and styles. We argue that future sarcasm research should take the broad scope of sarcasm into account.