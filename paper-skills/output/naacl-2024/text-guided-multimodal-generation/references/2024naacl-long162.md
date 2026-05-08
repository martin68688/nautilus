---
title: "Visually-Aware Context Modeling for News Image Captioning"
source: "https://aclanthology.org/2024.naacl-long.162/"
categories: ['llm-evaluation-summarization-argument-extraction', 'text-guided-multimodal-generation']
tags: ['image-captioning', 'news', 'context-modeling']
venue: "NAACL 2024"
tldr: "Enhances news image captioning with visually-aware context modeling, focusing on face-name co-occurrence."
---

# Visually-Aware Context Modeling for News Image Captioning

**Source**: [https://aclanthology.org/2024.naacl-long.162/](https://aclanthology.org/2024.naacl-long.162/)

**TLDR**: Enhances news image captioning with visually-aware context modeling, focusing on face-name co-occurrence.

## Abstract

AbstractNews Image Captioning aims to create captions from news articles and images, emphasizing the connection between textual context and visual elements. Recognizing the significance of human faces in news images and the face-name co-occurrence pattern in existing datasets, we propose a face-naming module for learning better name embeddings. Apart from names, which can be directly linked to an image area (faces), news image captions mostly contain context information that can only be found in the article. We design a retrieval strategy using CLIP to retrieve sentences that are semantically close to the image, mimicking human thought process of linking articles to images. Furthermore, to tackle the problem of the imbalanced proportion of article context and image context in captions, we introduce a simple yet effective method Contrasting with Language Model backbone (CoLaM) to the training pipeline. We conduct extensive experiments to demonstrate the efficacy of our framework. We out-perform the previous state-of-the-art (without external data) by 7.97/5.80 CIDEr scores on GoodNews/NYTimes800k. Our code is available at https://github.com/tingyu215/VACNIC.