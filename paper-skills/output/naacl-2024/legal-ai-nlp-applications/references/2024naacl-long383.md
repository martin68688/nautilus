---
title: "Reliability Estimation of News Media Sources: Birds of a Feather Flock Together"
source: "https://aclanthology.org/2024.naacl-long.383/"
categories: ['legal-ai-nlp-applications']
tags: ['reliability', 'news-media', 'network-analysis']
venue: "NAACL 2024"
tldr: "Proposes a method to estimate news source reliability based on citation network analysis."
---

# Reliability Estimation of News Media Sources: Birds of a Feather Flock Together

**Source**: [https://aclanthology.org/2024.naacl-long.383/](https://aclanthology.org/2024.naacl-long.383/)

**TLDR**: Proposes a method to estimate news source reliability based on citation network analysis.

## Abstract

AbstractEvaluating the reliability of news sources is a routine task for journalists and organizations committed to acquiring and disseminating accurate information.Recent research has shown that predicting sources’ reliability represents an important first-prior step in addressing additional challenges such as fake news detection and fact-checking.In this paper, we introduce a novel approach for source reliability estimation that leverages reinforcement learning strategies for estimating the reliability degree of news sources. Contrary to previous research, our proposed approach models the problem as the estimation of a reliability degree, and not a reliability label, based on how all the news media sources interact with each other on the Web.We validated the effectiveness of our method on a news media reliability dataset that is an order of magnitude larger than comparable existing datasets. Results show that the estimated reliability degrees strongly correlates with journalists-provided scores (Spearman=0.80) and can effectively predict reliability labels (macro-avg. F1 score=81.05).We release our implementation and dataset, aiming to provide a valuable resource for the NLP community working on information verification.