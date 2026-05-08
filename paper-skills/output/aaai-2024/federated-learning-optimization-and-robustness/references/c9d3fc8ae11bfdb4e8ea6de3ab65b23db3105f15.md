---
title: "FedCD: Federated Semi-Supervised Learning with Class Awareness Balance via Dual Teachers"
source: "https://www.semanticscholar.org/paper/c9d3fc8ae11bfdb4e8ea6de3ab65b23db3105f15"
categories: ['reinforcement-learning-bandits-planning-optimization', 'federated-learning-optimization-and-robustness']
tags: ['federated-learning', 'semi-supervised', 'class-imbalance', 'dual-teacher', 'medical-ai']
venue: "AAAI 2024"
tldr: "Introduces a federated semi-supervised learning method with dual teachers to balance class awareness and address data heterogeneity in medical diagnostics."
---

# FedCD: Federated Semi-Supervised Learning with Class Awareness Balance via Dual Teachers

**Source**: [https://www.semanticscholar.org/paper/c9d3fc8ae11bfdb4e8ea6de3ab65b23db3105f15](https://www.semanticscholar.org/paper/c9d3fc8ae11bfdb4e8ea6de3ab65b23db3105f15)

**TLDR**: Introduces a federated semi-supervised learning method with dual teachers to balance class awareness and address data heterogeneity in medical diagnostics.

## Abstract

Recent advancements in deep learning have greatly improved the efficiency of auxiliary medical diagnostics. However, concerns over patient privacy and data annotation costs restrict the viability of centralized training models. In response, federated semi-supervised learning has garnered substantial attention from medical institutions. However, it faces challenges arising from knowledge discrepancies among local clients and class imbalance in non-independent and identically distributed data. Existing methods like class balance adaptation for addressing class imbalance often overlook low-confidence yet valuable rare samples in unlabeled data and may compromise client privacy. To address these issues, we propose a novel framework with class awareness balance and dual teacher distillation called FedCD. FedCD introduces a global-local framework to balance and purify global and local knowledge. Additionally, we introduce a novel class awareness balance module to effectively explore potential rare classes and encourage balanced learning in unlabeled clients. Importantly, our approach prioritizes privacy protection by only exchanging network parameters during communication. Experimental results on two medical datasets under various settings demonstrate the effectiveness of FedCD. The code is available at https://github.com/YunzZ-Liu/FedCD.