---
title: "Signer Diversity-driven Data Augmentation for Signer-Independent Sign Language Translation"
source: "https://aclanthology.org/2024.findings-naacl.140/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'large-language-model-evaluation-augmentation']
tags: ['sign-language-translation', 'data-augmentation', 'signer-independence', 'multimodal']
venue: "NAACL 2024"
tldr: "Proposes a signer diversity-driven data augmentation method to improve the generalization of sign language translation systems to unseen signers."
---

# Signer Diversity-driven Data Augmentation for Signer-Independent Sign Language Translation

**Source**: [https://aclanthology.org/2024.findings-naacl.140/](https://aclanthology.org/2024.findings-naacl.140/)

**TLDR**: Proposes a signer diversity-driven data augmentation method to improve the generalization of sign language translation systems to unseen signers.

## Abstract

AbstractThe primary objective of sign language translation (SLT) is to transform sign language videos into natural sentences.A crucial challenge in this field is developing signer-independent SLT systems which requires models to generalize effectively to signers not encountered during training.This challenge is exacerbated by the limited diversity of signers in existing SLT datasets, which often results in suboptimal generalization capabilities of current models.Achieving robustness to unseen signers is essential for signer-independent SLT.However, most existing method relies on signer identity labels, which is often impractical and costly in real-world applications.To address this issue, we propose the Signer Diversity-driven Data Augmentation (SDDA) method that can achieve good generalization without relying on signer identity labels. SDDA comprises two data augmentation schemes. The first is data augmentation based on adversarial training, which aims to utilize the gradients of the model to generate adversarial examples. The second is data augmentation based on diffusion model, which focuses on using the advanced diffusion-based text guided image editing method to modify the appearances of the signer in images. The combination of the two strategies significantly enriches the diversity of signers in the training process.Moreover, we introduce a consistency loss and a discrimination loss to enhance the learning of signer-independent features.Our experimental results demonstrate our model significantly enhances the performance of SLT in the signer-independent setting, achieving state-of-the-art results without relying on signer identity labels.