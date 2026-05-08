---
title: "Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation"
source: "https://aclanthology.org/2024.naacl-long.2/"
categories: ['llm-alignment-safety-detoxification', 'text-guided-multimodal-generation']
tags: ['diffusion', 'text-generation', 'encoder-decoder']
venue: "NAACL 2024"
tldr: "Proposes a sequence-to-sequence text diffusion model using encoder-decoder transformers for discrete text generation."
---

# Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation

**Source**: [https://aclanthology.org/2024.naacl-long.2/](https://aclanthology.org/2024.naacl-long.2/)

**TLDR**: Proposes a sequence-to-sequence text diffusion model using encoder-decoder transformers for discrete text generation.

## Abstract

AbstractThe diffusion model, a new generative modeling paradigm, has achieved great success in image, audio, and video generation.However, considering the discrete categorical nature of the text, it is not trivial to extend continuous diffusion models to natural language. In this work, we propose SeqDiffuSeq, a text diffusion model, to approach sequence-to-sequence text generation with an encoder-decoder Transformer architecture.To improve the generation performance, SeqDiffuSeq is equipped with the self-conditioning technique and our newly proposed adaptive noise schedule technique. Self-conditioning enables SeqDiffuSeq to better use the predicted sequence information during the generation process.The adaptive noise schedule balances the difficulty of denoising across time steps at the token level.Experiment results illustrate the improved performance on five sequence-to-sequence generation tasks compared to other diffusion-based models regarding text quality and inference time.