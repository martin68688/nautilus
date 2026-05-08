---
title: "msLPCC: A Multimodal-Driven Scalable Framework for Deep LiDAR Point Cloud Compression"
source: "https://www.semanticscholar.org/paper/1c70efbcfea61beb47bee20c50d753e2db0fd303"
categories: ['llm-safety-adversarial-robustness', 'multimodal-time-series-foundation-models']
tags: ['lidar', 'point-cloud', 'compression', 'multimodal', 'autonomous-driving']
venue: "AAAI 2024"
tldr: "A multimodal-driven scalable framework for deep compression of large-scale, unevenly distributed LiDAR point clouds."
---

# msLPCC: A Multimodal-Driven Scalable Framework for Deep LiDAR Point Cloud Compression

**Source**: [https://www.semanticscholar.org/paper/1c70efbcfea61beb47bee20c50d753e2db0fd303](https://www.semanticscholar.org/paper/1c70efbcfea61beb47bee20c50d753e2db0fd303)

**TLDR**: A multimodal-driven scalable framework for deep compression of large-scale, unevenly distributed LiDAR point clouds.

## Abstract

LiDAR sensors are widely used in autonomous driving, and the growing storage and transmission demands have made LiDAR point cloud compression (LPCC) a hot research topic. To address the challenges posed by the large-scale and uneven-distribution (spatial and categorical) of LiDAR point data, this paper presents a new multimodal-driven scalable LPCC framework. For the large-scale challenge, we decouple the original LiDAR data into multi-layer point subsets, compress and transmit each layer separately, so as to ensure the reconstruction quality requirement under different scenarios. For the uneven-distribution challenge, we extract, align, and fuse heterologous feature representations, including point modality with position information, depth modality with spatial distance information, and segmentation modality with category information. Extensive experimental results on the benchmark SemanticKITTI database validate that our method outperforms 14 recent representative LPCC methods.