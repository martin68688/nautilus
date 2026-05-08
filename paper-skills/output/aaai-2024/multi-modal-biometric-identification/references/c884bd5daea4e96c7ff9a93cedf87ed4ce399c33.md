---
title: "Multi-Modal Disordered Representation Learning Network for Description-Based Person Search"
source: "https://www.semanticscholar.org/paper/c884bd5daea4e96c7ff9a93cedf87ed4ce399c33"
categories: ['graph-learning-clustering-multiview-federated', 'multi-modal-biometric-identification']
tags: ['person-search', 'multi-modal', 'disordered-representation', 'description-based']
venue: "AAAI 2024"
tldr: "Learns discriminative representations for description-based person search via a multi-modal disordered representation network."
---

# Multi-Modal Disordered Representation Learning Network for Description-Based Person Search

**Source**: [https://www.semanticscholar.org/paper/c884bd5daea4e96c7ff9a93cedf87ed4ce399c33](https://www.semanticscholar.org/paper/c884bd5daea4e96c7ff9a93cedf87ed4ce399c33)

**TLDR**: Learns discriminative representations for description-based person search via a multi-modal disordered representation network.

## Abstract

Description-based person search aims to retrieve images of the target identity via textual descriptions. One of the challenges for this task is to extract discriminative representation from images and descriptions. Most existing methods apply the part-based split method or external models to explore the fine-grained details of local features, which ignore the global relationship between partial information and cause network instability. To overcome these issues, we propose a Multi-modal Disordered Representation Learning Network (MDRL) for description-based person search to fully extract the visual and textual representations. Specifically, we design a Cross-modality Global Feature Learning Architecture to learn the global features from the two modalities and meet the demand of the task. Based on our global network, we introduce a Disorder Local Learning Module to explore local features by a disordered reorganization strategy from both visual and textual aspects and enhance the robustness of the whole network. Besides, we introduce a Cross-modality Interaction Module to guide the two streams to extract visual or textual representations considering the correlation between modalities. Extensive experiments are conducted on two public datasets, and the results show that our method outperforms the state-of-the-art methods on CUHK-PEDES and ICFG-PEDES datasets and achieves superior performance.