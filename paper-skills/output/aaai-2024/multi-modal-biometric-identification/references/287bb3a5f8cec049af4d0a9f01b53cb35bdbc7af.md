---
title: "Text-Based Occluded Person Re-identification via Multi-Granularity Contrastive Consistency Learning"
source: "https://www.semanticscholar.org/paper/287bb3a5f8cec049af4d0a9f01b53cb35bdbc7af"
categories: ['multi-modal-biometric-identification', 'public-safety-ml-prediction-intervention']
tags: ['person-re-identification', 'text-based', 'occlusion-robust']
venue: "AAAI 2024"
tldr: "Uses multi-granularity contrastive learning for text-based occluded person re-identification."
---

# Text-Based Occluded Person Re-identification via Multi-Granularity Contrastive Consistency Learning

**Source**: [https://www.semanticscholar.org/paper/287bb3a5f8cec049af4d0a9f01b53cb35bdbc7af](https://www.semanticscholar.org/paper/287bb3a5f8cec049af4d0a9f01b53cb35bdbc7af)

**TLDR**: Uses multi-granularity contrastive learning for text-based occluded person re-identification.

## Abstract

Text-based Person Re-identification (T-ReID), which aims at retrieving a specific pedestrian image from a collection of images via text-based information, has received significant attention. However, previous research has overlooked a challenging yet practical form of T-ReID: dealing with image galleries mixed with occluded and inconsistent personal visuals, instead of ideal visuals with a full-body and clear view. Its major challenges lay in the insufficiency of benchmark datasets and the enlarged semantic gap incurred by arbitrary occlusions and modality gap between text description and visual representation of the target person. To alleviate these issues, we first design an Occlusion Generator (OGor) for the automatic generation of artificial occluded images from generic surveillance images. Then, a fine-granularity token selection mechanism is proposed to minimize the negative impact of occlusion for robust feature learning, and a novel multi-granularity contrastive consistency alignment framework is designed to leverage intra-/inter-granularity of visual-text representations for semantic alignment of occluded visuals and query texts. Experimental results demonstrate that our method exhibits superior performance. We believe this work could inspire the community to investigate more dedicated designs for implementing T-ReID in real-world scenarios. The source code is available at https://github.com/littlexinyi/MGCC.