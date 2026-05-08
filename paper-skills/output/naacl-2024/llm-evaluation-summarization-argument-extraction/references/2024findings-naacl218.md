---
title: "Select and Summarize: Scene Saliency for Movie Script Summarization"
source: "https://aclanthology.org/2024.findings-naacl.218/"
categories: ['llm-evaluation-summarization-argument-extraction', 'metaphor-analysis-political-framing']
tags: ['summarization', 'movie-scripts', 'scene-saliency', 'long-form']
venue: "NAACL 2024"
tldr: "Uses scene saliency to select key scenes for abstractive summarization of long movie scripts."
---

# Select and Summarize: Scene Saliency for Movie Script Summarization

**Source**: [https://aclanthology.org/2024.findings-naacl.218/](https://aclanthology.org/2024.findings-naacl.218/)

**TLDR**: Uses scene saliency to select key scenes for abstractive summarization of long movie scripts.

## Abstract

AbstractAbstractive summarization for long-form narrative texts such as movie scripts is challenging due to the computational and memory constraints of current language models. A movie script typically comprises a large number of scenes; however, only a fraction of these scenes are salient, i.e., important for understanding the overall narrative. The salience of a scene can be operationalized by considering it as salient if it is mentioned in the summary. Automatically identifying salient scenes is difficult due to the lack of suitable datasets. In this work, we introduce a scene saliency dataset that consists of human-annotated salient scenes for 100 movies. We propose a two-stage abstractive summarization approach which first identifies the salient scenes in script and then generates a summary using only those scenes. Using QA-based evaluation, we show that our model outperforms previous state-of-the-art summarization methods and reflects the information content of a movie more accurately than a model that takes the whole movie script as input.