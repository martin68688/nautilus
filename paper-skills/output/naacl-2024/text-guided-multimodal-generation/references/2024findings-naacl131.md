---
title: "PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck"
source: "https://aclanthology.org/2024.findings-naacl.131/"
categories: ['zero-shot-multimodal-large-language-models', 'text-guided-multimodal-generation']
tags: ['vision-language', 'explainable-ai', 'image-classification', 'language-bottleneck']
venue: "NAACL 2024"
tldr: "PEEB is a part-based image classifier with an explainable and editable language bottleneck that improves fine-grained classification, especially for new or rare classes, by using textual part descriptions."
---

# PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck

**Source**: [https://aclanthology.org/2024.findings-naacl.131/](https://aclanthology.org/2024.findings-naacl.131/)

**TLDR**: PEEB is a part-based image classifier with an explainable and editable language bottleneck that improves fine-grained classification, especially for new or rare classes, by using textual part descriptions.

## Abstract

AbstractCLIP-based classifiers rely on the prompt containing a class name that is known to the text encoder. Therefore, they perform poorly on new classes or the classes whose names rarely appear on the Internet (e.g., scientific names of birds). For fine-grained classification, we propose PEEB – an explainable and editable classifier to (1) express the class name into a set of text descriptors that describe the visual parts of that class; and (2) match the embeddings of the detected parts to their textual descriptors in each class to compute a logit score for classification. In a zero-shot setting where the class names are unknown, PEEB outperforms CLIP by a huge margin (∼10× in top-1 accuracy). Compared to part-based classifiers, PEEB is not only the state-of-the-art (SOTA) on the supervised-learning setting (88.80% and 92.20% accuracy on CUB-200 and Stanford Dogs-120, respectively) but also the first to enable users to edit the text descriptors to form a new classifier without any re-training. Compared to concept bottleneck models, PEEB is also the SOTA in both zero-shot and supervised-learning settings.