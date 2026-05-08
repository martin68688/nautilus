---
title: "Knowledge-Enhanced Historical Document Segmentation and Recognition"
source: "https://www.semanticscholar.org/paper/96d2b9bc85ef8cbd441edb5eaca704f65c2d57ab"
categories: ['reinforcement-learning-bandits-planning-optimization', 'historical-document-ocr-enhancement']
tags: ['historical-document-ocr', 'knowledge-graph', 'data-augmentation']
venue: "AAAI 2024"
tldr: "Enhances historical document OCR by integrating knowledge graphs for data augmentation and a multi-task learning model for joint segmentation and recognition."
---

# Knowledge-Enhanced Historical Document Segmentation and Recognition

**Source**: [https://www.semanticscholar.org/paper/96d2b9bc85ef8cbd441edb5eaca704f65c2d57ab](https://www.semanticscholar.org/paper/96d2b9bc85ef8cbd441edb5eaca704f65c2d57ab)

**TLDR**: Enhances historical document OCR by integrating knowledge graphs for data augmentation and a multi-task learning model for joint segmentation and recognition.

## Abstract

Optical Character Recognition (OCR) of historical document images remains a challenging task because of the distorted input images, extensive number of uncommon characters, and the scarcity of labeled data, which impedes modern deep learning-based OCR techniques from achieving good recognition accuracy. Meanwhile, there exists a substantial amount of expert knowledge that can be utilized in this task. However, such knowledge is usually complicated and could only be accurately expressed with formal languages such as first-order logic (FOL), which is difficult to be directly integrated into deep learning models. This paper proposes KESAR, a novel Knowledge-Enhanced Document Segmentation And Recognition method for historical document images based on the Abductive Learning (ABL) framework. The segmentation and recognition models are enhanced by incorporating background knowledge for character extraction and prediction, followed by an efficient joint optimization of both models. We validate the effectiveness of KESAR on historical document datasets. The experimental results demonstrate that our method can simultaneously utilize knowledge-driven reasoning and data-driven learning, which outperforms the current state-of-the-art methods.