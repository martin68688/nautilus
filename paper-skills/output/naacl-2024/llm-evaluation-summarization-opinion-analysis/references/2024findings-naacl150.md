---
title: "Product Description and QA Assisted Self-Supervised Opinion Summarization"
source: "https://aclanthology.org/2024.findings-naacl.150/"
categories: ['llm-evaluation-summarization-opinion-analysis', 'large-language-model-evaluation-augmentation']
tags: ['opinion-summarization', 'self-supervised', 'e-commerce']
venue: "NAACL 2024"
tldr: "Proposes a self-supervised opinion summarization method for e-commerce using product descriptions and question-answers."
---

# Product Description and QA Assisted Self-Supervised Opinion Summarization

**Source**: [https://aclanthology.org/2024.findings-naacl.150/](https://aclanthology.org/2024.findings-naacl.150/)

**TLDR**: Proposes a self-supervised opinion summarization method for e-commerce using product descriptions and question-answers.

## Abstract

AbstractIn e-commerce, opinion summarization is the process of summarizing the consensus opinions found in product reviews. However, the potential of additional sources such as product description and question-answers (QA) has been considered less often. Moreover, the absence of any supervised training data makes this task challenging. To address this, we propose a novel synthetic dataset creation (SDC) strategy that leverages information from reviews as well as additional sources for selecting one of the reviews as a pseudo-summary to enable supervised training. Our Multi-Encoder Decoder framework for Opinion Summarization (MEDOS) employs a separate encoder for each source, enabling effective selection of information while generating the summary. For evaluation, due to the unavailability of test sets with additional sources, we extend the Amazon, Oposum+, and Flipkart test sets and leverage ChatGPT to annotate summaries. Experiments across nine test sets demonstrate that the combination of our SDC approach and MEDOS model achieves on average a 14.5% improvement in ROUGE-1 F1 over the SOTA. Moreover, comparative analysis underlines the significance of incorporating additional sources for generating more informative summaries. Human evaluations further indicate that MEDOS scores relatively higher in coherence and fluency with 0.41 and 0.5 (−1 to 1) respectively, compared to existing models. To the best of our knowledge, we are the first to generate opinion summaries leveraging additional sources in a self-supervised setting.