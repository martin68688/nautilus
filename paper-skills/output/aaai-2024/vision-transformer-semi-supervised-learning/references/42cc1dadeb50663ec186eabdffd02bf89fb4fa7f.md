---
title: "Novel Class Discovery in Chest X-rays via Paired Images and Text"
source: "https://www.semanticscholar.org/paper/42cc1dadeb50663ec186eabdffd02bf89fb4fa7f"
categories: ['vision-transformer-semi-supervised-learning', 'causal-inference-graph-machine-learning']
tags: ['novel-class-discovery', 'chest-xray', 'vision-language']
venue: "AAAI 2024"
tldr: "Discovers novel classes in chest X-rays using paired image-text data and knowledge from known classes."
---

# Novel Class Discovery in Chest X-rays via Paired Images and Text

**Source**: [https://www.semanticscholar.org/paper/42cc1dadeb50663ec186eabdffd02bf89fb4fa7f](https://www.semanticscholar.org/paper/42cc1dadeb50663ec186eabdffd02bf89fb4fa7f)

**TLDR**: Discovers novel classes in chest X-rays using paired image-text data and knowledge from known classes.

## Abstract

Novel class discover(NCD) aims to identify new classes undefined during model training phase with the help of knowledge of known classes. Many methods have been proposed and notably boosted performance of NCD in natural images. However, there has been no work done in discovering new classes based on medical images and disease categories, which is crucial for understanding and diagnosing specific diseases. Moreover, most of the existing methods only utilize information from image modality and use labels as the only supervisory information. In this paper, we propose a multi-modal novel class discovery method based on paired images and text, inspired by the low classification accuracy of chest X-ray images and the relatively higher accuracy of the paired text. Specifically, we first pretrain the image encoder and text encoder with multi-modal contrastive learning on the entire dataset and then we generate pseudo-labels separately on the image branch and text branch. We utilize intra-modal consistency to assess the quality of pseudo-labels and adjust the weights of the pseudo-labels from both branches to generate the ultimate pseudo-labels for training. Experiments on eight subset splits of MIMIC-CXR-JPG dataset show that our method improves the clustering performance of unlabeled classes by about 10% on average compared to state-of-the-art methods. Code is available at: https://github.com/zzzzzzzzjy/MMNCD-main.