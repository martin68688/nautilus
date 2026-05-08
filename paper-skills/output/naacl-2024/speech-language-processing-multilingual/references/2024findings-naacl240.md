---
title: "CM-TTS: Enhancing Real Time Text-to-Speech Synthesis Efficiency through Weighted Samplers and Consistency Models"
source: "https://aclanthology.org/2024.findings-naacl.240/"
categories: ['human-llm-opinion-dynamics-moderation', 'speech-language-processing-multilingual']
tags: ['text-to-speech', 'consistency-models', 'efficiency']
venue: "NAACL 2024"
tldr: "Enhances real-time TTS efficiency using weighted samplers and consistency models."
---

# CM-TTS: Enhancing Real Time Text-to-Speech Synthesis Efficiency through Weighted Samplers and Consistency Models

**Source**: [https://aclanthology.org/2024.findings-naacl.240/](https://aclanthology.org/2024.findings-naacl.240/)

**TLDR**: Enhances real-time TTS efficiency using weighted samplers and consistency models.

## Abstract

AbstractNeural Text-to-Speech (TTS) systems find broad applications in voice assistants, e-learning, and audiobook creation. The pursuit of modern models, like Diffusion Models (DMs), holds promise for achieving high-fidelity, real-time speech synthesis. Yet, the efficiency of multi-step sampling in Diffusion Models presents challenges. Efforts have been made to integrate GANs with DMs, speeding up inference by approximating denoising distributions, but this introduces issues with model convergence due to adversarial training. To overcome this, we introduce CM-TTS, a novel architecture grounded in consistency models (CMs). Drawing inspiration from continuous-time diffusion models, CM-TTS achieves top-quality speech synthesis in fewer steps without adversarial training or pre-trained model dependencies. We further design weighted samplers to incorporate different sampling positions into model training with dynamic probabilities, ensuring unbiased learning throughout the entire training process. We present a real-time mel-spectrogram generation consistency model, validated through comprehensive evaluations. Experimental results underscore CM-TTS’s superiority over existing single-step speech synthesis systems, representing a significant advancement in the field.