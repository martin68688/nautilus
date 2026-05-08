---
title: "SNN-PDE: Learning Dynamic PDEs from Data with Simplicial Neural Networks"
source: "https://www.semanticscholar.org/paper/b2928c9c56daf45d01b8df307cb7baf81ffbc7bc"
categories: ['topological-geometric-data-representation']
tags: ['pde-learning', 'simplicial-neural-networks', 'dynamic-systems']
venue: "AAAI 2024"
tldr: "Uses simplicial neural networks to learn dynamic partial differential equations from data."
---

# SNN-PDE: Learning Dynamic PDEs from Data with Simplicial Neural Networks

**Source**: [https://www.semanticscholar.org/paper/b2928c9c56daf45d01b8df307cb7baf81ffbc7bc](https://www.semanticscholar.org/paper/b2928c9c56daf45d01b8df307cb7baf81ffbc7bc)

**TLDR**: Uses simplicial neural networks to learn dynamic partial differential equations from data.

## Abstract

Dynamics of many complex systems, from weather and climate to spread of infectious diseases, can be described by partial differential equations (PDEs). Such PDEs involve unknown function(s), partial derivatives, and typically multiple independent variables. The traditional numerical methods for solving PDEs assume that the data are observed on a regular grid. However, in many applications, for example, weather and air pollution monitoring delivered by the arbitrary located weather stations of the National Weather Services, data records are irregularly spaced. Furthermore, in problems involving prediction analytics such as forecasting wildfire smoke plumes, the primary focus may be on a set of irregular locations associated with urban development. In recent years, deep learning (DL) methods and, in particular, graph neural networks (GNNs) have emerged as a new promising tool that can complement traditional PDE solvers in scenarios of the irregular spaced data, contributing to the newest research trend of physics informed machine learning (PIML). However, most existing PIML methods tend to be limited in their ability to describe higher dimensional structural properties exhibited by real world phenomena, especially, ones that live on manifolds. To address this fundamental challenge, we bring the elements of the Hodge theory and, in particular, simplicial convolution defined on the Hodge Laplacian to the emerging nexus of DL and PDEs. In contrast to conventional Laplacian and the associated convolution operation, the simplicial convolution allows us to rigorously describe diffusion across higher order structures and to better approximate the complex underlying topology and geometry of the data. The new approach, Simplicial Neural Networks for Partial Differential Equations (SNN PDE) offers a computationally efficient yet effective solution for time dependent PDEs. Our studies of a broad range of synthetic data and wildfire processes demonstrate that SNN PDE improves upon state of the art baselines in handling unstructured grids and irregular time intervals of complex physical systems and offers competitive forecasting capabilities for weather and air quality forecasting.