---
title: "Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt"
source: "https://aclanthology.org/2024.naacl-long.268/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'text-guided-multimodal-generation']
tags: ['singing-synthesis', 'controllable-generation', 'natural-language-prompt']
venue: "NAACL 2024"
tldr: "Prompt-Singer is a controllable singing-voice-synthesis model that uses natural language prompts to manipulate style attributes."
---

# Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt

**Source**: [https://aclanthology.org/2024.naacl-long.268/](https://aclanthology.org/2024.naacl-long.268/)

**TLDR**: Prompt-Singer is a controllable singing-voice-synthesis model that uses natural language prompts to manipulate style attributes.

## Abstract

AbstractRecent singing-voice-synthesis (SVS) methods have achieved remarkable audio quality and naturalness, yet they lack the capability to control the style attributes of the synthesized singing explicitly. We propose Prompt-Singer, the first SVS method that enables attribute controlling on singer gender, vocal range and volume with natural language. We adopt a model architecture based on a decoder-only transformer with a multi-scale hierarchy, and design a range-melody decoupled pitch representation that enables text-conditioned vocal range control while keeping melodic accuracy. Furthermore, we explore various experiment settings, including different types of text representations, text encoder fine-tuning, and introducing speech data to alleviate data scarcity, aiming to facilitate further research. Experiments show that our model achieves favorable controlling ability and audio quality. Audio samples are available at http://prompt-singer.github.io .