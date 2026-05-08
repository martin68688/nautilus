---
title: "Polarity Calibration for Opinion Summarization"
source: "https://aclanthology.org/2024.naacl-long.291/"
categories: ['llm-evaluation-summarization-argument-extraction', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['opinion-summarization', 'polarity', 'calibration']
venue: "NAACL 2024"
tldr: "This work introduces a polarity calibration method for opinion summarization to better present divergent and conflicting opinions."
---

# Polarity Calibration for Opinion Summarization

**Source**: [https://aclanthology.org/2024.naacl-long.291/](https://aclanthology.org/2024.naacl-long.291/)

**TLDR**: This work introduces a polarity calibration method for opinion summarization to better present divergent and conflicting opinions.

## Abstract

AbstractOpinion summarization is automatically generating summaries from a variety of subjective information, such as product reviews or political opinions. The challenge of opinions summarization lies in presenting divergent or even conflicting opinions. We conduct an analysis of previous summarization models, which reveals their inclination to amplify the polarity bias, emphasizing the majority opinions while ignoring the minority opinions. To address this issue and make the summarizer express both sides of opinions, we introduce the concept of polarity calibration, which aims to align the polarity of output summary with that of input text. Specifically, we develop a reinforcement training approach for polarity calibration. This approach feeds the polarity distance between output summary and input text as reward into the summarizer, and also balance polarity calibration with content preservation and language naturality. We evaluate our Polarity Calibration model (PoCa) on two types of opinions summarization tasks: summarizing product reviews and political opinions articles. Automatic and human evaluation demonstrate that our approach can mitigate the polarity mismatch between output summary and input text, as well as maintain the content semantic and language quality.