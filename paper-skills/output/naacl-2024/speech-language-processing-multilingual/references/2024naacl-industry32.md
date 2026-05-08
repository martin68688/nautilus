---
title: "AMA-LSTM: Pioneering Robust and Fair Financial Audio Analysis for Stock Volatility Prediction"
source: "https://aclanthology.org/2024.naacl-industry.32/"
categories: ['financial-risk-narrative-analysis', 'speech-language-processing-multilingual']
tags: ['financial-audio', 'stock-prediction', 'multimodal', 'robustness']
venue: "NAACL 2024"
tldr: "Presents a robust and fair multimodal model for stock volatility prediction using audio from earnings calls."
---

# AMA-LSTM: Pioneering Robust and Fair Financial Audio Analysis for Stock Volatility Prediction

**Source**: [https://aclanthology.org/2024.naacl-industry.32/](https://aclanthology.org/2024.naacl-industry.32/)

**TLDR**: Presents a robust and fair multimodal model for stock volatility prediction using audio from earnings calls.

## Abstract

AbstractStock volatility prediction is an important task in the financial industry. Recent multimodal methods have shown advanced results by combining text and audio information, such as earnings calls. However, these multimodal methods have faced two drawbacks. First, they often fail to yield reliable models and overfit the data due to their absorption of stochastic information from the stock market. Moreover, using multimodal models to predict stock volatility suffers from gender bias and lacks an efficient way to eliminate such bias. To address these aforementioned problems, we use adversarial training to generate perturbations that simulate the inherent stochasticity and bias, by creating areas resistant to random information around the input space to improve model robustness and fairness. Our comprehensive experiments on two real-world financial audio datasets reveal that this method exceeds the performance of current state-of-the-art solution. This confirms the value of adversarial training in reducing stochasticity and bias for stock volatility prediction tasks.