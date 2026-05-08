---
title: "PLATYPUS: Progressive Local Surface Estimator for Arbitrary-Scale Point Cloud Upsampling"
source: "https://www.semanticscholar.org/paper/a3f734d86bd0a3ebf9b8c33661c6f3ac8cd4acca"
categories: ['topological-geometric-data-representation']
tags: ['point-cloud-upsampling', '3d-reconstruction', 'progressive-learning']
venue: "AAAI 2024"
tldr: "PLATYPUS is a progressive local surface estimator for arbitrary-scale point cloud upsampling that improves density and uniformity in 3D data."
---

# PLATYPUS: Progressive Local Surface Estimator for Arbitrary-Scale Point Cloud Upsampling

**Source**: [https://www.semanticscholar.org/paper/a3f734d86bd0a3ebf9b8c33661c6f3ac8cd4acca](https://www.semanticscholar.org/paper/a3f734d86bd0a3ebf9b8c33661c6f3ac8cd4acca)

**TLDR**: PLATYPUS is a progressive local surface estimator for arbitrary-scale point cloud upsampling that improves density and uniformity in 3D data.

## Abstract

3D point clouds are increasingly vital for applications like autonomous driving and robotics, yet the raw data captured by sensors often suffer from noise and sparsity, creating challenges for downstream tasks. Consequently, point cloud upsampling becomes essential for improving density and uniformity, with recent approaches showing promise by projecting randomly generated query points onto the underlying surface of sparse point clouds. However, these methods often result in outliers, non-uniformity, and difficulties in handling regions with high curvature and intricate structures. In this work, we address these challenges by introducing the Progressive Local Surface Estimator (PLSE), which more effectively captures local features in complex regions through a curvature-based sampling technique that selectively targets high-curvature areas. Additionally, we incorporate a curriculum learning strategy that leverages the curvature distribution within the point cloud to naturally assess the sample difficulty, enabling curriculum learning on point cloud data for the first time. The experimental results demonstrate that our approach significantly outperforms existing methods, achieving high-quality, dense point clouds with superior accuracy and detail.