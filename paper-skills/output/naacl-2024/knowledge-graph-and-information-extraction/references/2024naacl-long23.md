---
title: "Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision"
source: "https://aclanthology.org/2024.naacl-long.23/"
categories: ['knowledge-graph-and-information-extraction', 'legal-ai-nlp-applications', 'knowledge-graph-hallucination-reduction']
tags: ['multimodal', 'hallucination', 'self-feedback']
venue: "NAACL 2024"
tldr: "Mitigates multimodal hallucination in LMMs through a self-feedback guided revision mechanism that corrects initial responses."
---

# Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision

**Source**: [https://aclanthology.org/2024.naacl-long.23/](https://aclanthology.org/2024.naacl-long.23/)

**TLDR**: Mitigates multimodal hallucination in LMMs through a self-feedback guided revision mechanism that corrects initial responses.

## Abstract

AbstractLarge multimodal models suffer from multimodal hallucination, where they provide incorrect responses misaligned with the given visual information. Recent works have conjectured that one of the reasons behind multimodal hallucination is due to the vision encoder failing to ground on the image properly. To mitigate this issue, we propose a novel approach that leverages self-feedback as visual cues. Building on this approach, we introduce Volcano, a multimodal self-feedback guided revision model. Volcano generates natural language feedback to its initial response based on the provided visual information and utilizes this feedback to self-revise its initial response. Volcano effectively reduces multimodal hallucination and achieves state-of-the-art on MMHal-Bench, POPE, and GAVIE. It also improves on general multimodal abilities and outperforms previous models on MM-Vet and MMBench. Through qualitative analysis, we show that Volcano’s feedback is properly grounded on the image than the initial response. This indicates that Volcano can provide itself with richer visual information through feedback generation, leading to self-correct hallucinations. We publicly release our model, data, and code at https://github.com/kaistAI/Volcanogithub.com/kaistAI/Volcano