---
title: "MusiLingo: Bridging Music and Text with Pre-trained Language Models for Music Captioning and Query Response"
source: "https://aclanthology.org/2024.findings-naacl.231/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'text-guided-multimodal-generation']
tags: ['multimodal', 'music', 'language-model', 'captioning']
venue: "NAACL 2024"
tldr: "A system (MusiLingo) bridging music and text with pre-trained language models for music captioning and query response."
---

# MusiLingo: Bridging Music and Text with Pre-trained Language Models for Music Captioning and Query Response

**Source**: [https://aclanthology.org/2024.findings-naacl.231/](https://aclanthology.org/2024.findings-naacl.231/)

**TLDR**: A system (MusiLingo) bridging music and text with pre-trained language models for music captioning and query response.

## Abstract

AbstractLarge Language Models (LLMs) have shown immense potential in multimodal applications, yet the convergence of textual and musical domains remains not well-explored. To address this gap, we present MusiLingo, a novel system for music caption generation and music-related query responses. MusiLingo employs a single projection layer to align music representations from the pre-trained frozen music audio model MERT (CITATION) with a frozen LLM, bridging the gap between music audio and textual contexts. We train it on an extensive music caption dataset and fine-tune it with instructional data. Due to the scarcity of high-quality music Q&A datasets, we created the MusicInstruct (MI) dataset from captions in the MusicCaps datasets, tailored for open-ended music inquiries. Empirical evaluations demonstrate its competitive performance in generating music captions and composing music-related Q&A pairs. Our introduced dataset enables notable advancements beyond previous ones.