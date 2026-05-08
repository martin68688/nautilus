---
title: "SwiftPillars: High-Efficiency Pillar Encoder for Lidar-Based 3D Detection"
source: "https://www.semanticscholar.org/paper/fe64b08c16231505abdb1b5cf25ffcffbcf9ccff"
categories: ['light-field-microscopy-image-reconstruction']
tags: ['3d-detection', 'lidar', 'autonomous-driving', 'efficiency']
venue: "AAAI 2024"
tldr: "Designs a high-efficiency pillar encoder for Lidar-based 3D detection, simplifying architecture for easier deployment in autonomous driving."
---

# SwiftPillars: High-Efficiency Pillar Encoder for Lidar-Based 3D Detection

**Source**: [https://www.semanticscholar.org/paper/fe64b08c16231505abdb1b5cf25ffcffbcf9ccff](https://www.semanticscholar.org/paper/fe64b08c16231505abdb1b5cf25ffcffbcf9ccff)

**TLDR**: Designs a high-efficiency pillar encoder for Lidar-based 3D detection, simplifying architecture for easier deployment in autonomous driving.

## Abstract

Lidar-based 3D Detection is one of the significant components of Autonomous Driving. However, current methods over-focus on improving the performance of 3D Lidar perception, which causes the architecture of networks becoming complicated and hard to deploy. Thus, the methods are difficult to apply in Autonomous Driving for real-time processing. In this paper, we propose a high-efficiency network, SwiftPillars, which includes Swift Pillar Encoder (SPE) and Multi-scale Aggregation Decoder (MAD). The SPE is constructed by a concise Dual-attention Module with lightweight operators. The Dual-attention Module utilizes feature pooling, matrix multiplication, etc. to speed up point-wise and channel-wise attention extraction and fusion. The MAD interconnects multiple scale features extracted by SPE with minimal computational cost to leverage performance. In our experiments, our proposal accomplishes 61.3% NDS and 53.2% mAP in nuScenes dataset. In addition, we evaluate inference time on several platforms (P4, T4, A2, MLU370, RTX3080), where SwiftPillars achieves up to 13.3ms (75FPS) on NVIDIA Tesla T4. Compared with PointPillars, SwiftPillars is on average 26.58% faster in inference speed with equivalent GPUs and a higher mAP of approximately 3.2% in the nuScenes dataset.