---
title: "SumCSE: Summary as a transformation for Contrastive Learning"
source: "https://aclanthology.org/2024.findings-naacl.227/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['contrastive-learning', 'summarization', 'data-augmentation']
venue: "NAACL 2024"
tldr: "Proposes using generated summaries as a data augmentation transformation for contrastive learning of sentence embeddings."
---

# SumCSE: Summary as a transformation for Contrastive Learning

**Source**: [https://aclanthology.org/2024.findings-naacl.227/](https://aclanthology.org/2024.findings-naacl.227/)

**TLDR**: Proposes using generated summaries as a data augmentation transformation for contrastive learning of sentence embeddings.

## Abstract

AbstractSentence embedding models are typically trained using contrastive learning (CL), either using human annotations directly or by repurposing other annotated datasets. In this work, we explore the recently introduced paradigm of generating CL data using generative language models (LM). In CL for computer vision (CV), compositional transformations (series of operations applied over an image. e.g. cropping + color distortion) which modify the input/image to retain minimal information were shown to be very effective. We show that composition of a ‘Summary’ transformation with diverse paraphrasing/contradicting transformations accomplishes the same and works very well in CL for sentence embeddings. Our final generated dataset (using Vicuna-13B) significantly outperforms the previous best unsupervised method (using ChatGPT) by 1.8 points, and SimCSE, a strong supervised baseline by 0.3 points on the semantic text similarity (STS) benchmark.