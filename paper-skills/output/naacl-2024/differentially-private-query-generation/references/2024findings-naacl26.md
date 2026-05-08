---
title: "VOLTA: Improving Generative Diversity by Variational Mutual Information Maximizing Autoencoder"
source: "https://aclanthology.org/2024.findings-naacl.26/"
categories: ['continual-learning-memory-transfer-llms', 'differentially-private-query-generation']
tags: ['generative-diversity', 'variational-autoencoder', 'mutual-information']
venue: "NAACL 2024"
tldr: "Improves generative diversity in language models via a variational mutual information maximizing autoencoder."
---

# VOLTA: Improving Generative Diversity by Variational Mutual Information Maximizing Autoencoder

**Source**: [https://aclanthology.org/2024.findings-naacl.26/](https://aclanthology.org/2024.findings-naacl.26/)

**TLDR**: Improves generative diversity in language models via a variational mutual information maximizing autoencoder.

## Abstract

AbstractThe natural language generation domain has witnessed great success thanks to Transformer models. Although they have achieved state-of-the-art generative quality, they often neglect generative diversity. Prior attempts to tackle this issue suffer from either low model capacity or over-complicated architectures. Some recent methods employ the VAE framework to enhance diversity, but their latent variables fully depend on the input context, restricting exploration of the latent space. In this paper, we introduce VOLTA, a framework that elevates generative diversity by bridging Transformer with VAE via a more effective cross-attention-based connection, departing from conventional embedding concatenation or summation. Additionally, we propose integrating InfoGAN-style latent codes to enable input-independent variability, further diversifying the generation. Moreover, our framework accommodates discrete inputs alongside its existing support for continuous inputs. We perform comprehensive experiments with two types of Transformers on six datasets from three different NLG tasks to show that our approach can significantly improve generative diversity while maintaining generative quality.