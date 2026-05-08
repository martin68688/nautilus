---
title: "Teaching a Multilingual Large Language Model to Understand Multilingual Speech via Multi-Instructional Training"
source: "https://aclanthology.org/2024.findings-naacl.52/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'speech-language-processing-multilingual']
tags: ['speech', 'multilingual', 'instruction-tuning']
venue: "NAACL 2024"
tldr: "Teaches a multilingual LLM to understand speech via multi-instructional training."
---

# Teaching a Multilingual Large Language Model to Understand Multilingual Speech via Multi-Instructional Training

**Source**: [https://aclanthology.org/2024.findings-naacl.52/](https://aclanthology.org/2024.findings-naacl.52/)

**TLDR**: Teaches a multilingual LLM to understand speech via multi-instructional training.

## Abstract

AbstractRecent advancements in language modeling have led to the emergenceof Large Language Models (LLMs) capable ofvarious natural language processing tasks.Despite their success in text-based tasks, applying LLMs to the speech domainremains limited and challenging. This paper presents BLOOMZMMS, a novel modelthat integrates a multilingual LLM with a multilingual speech encoder,aiming to harness the capabilities of LLMs for speech recognition and beyond.Utilizing a multi-instructional training approach, we demonstrate the transferabilityof linguistic knowledge from the text to the speech modality.Our experiments, conducted on 1900 hours of transcribed data from 139 languages,establish that a multilingual speech representation can be effectivelylearned and aligned with a multilingual LLM. While this learned representationinitially shows limitations in task generalization, we address this issue bygenerating synthetic targets in a multi-instructional style.Our zero-shot evaluation results confirm the robustness of our approach acrossmultiple tasks, including speech translation and multilingual spoken languageunderstanding, thereby opening new avenues for applying LLMs in the speech domain.