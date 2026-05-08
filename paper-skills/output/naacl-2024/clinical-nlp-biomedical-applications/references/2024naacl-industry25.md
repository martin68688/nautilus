---
title: "Multimodal Contextual Dialogue Breakdown Detection for Conversational AI Models"
source: "https://aclanthology.org/2024.naacl-industry.25/"
categories: ['clinical-nlp-biomedical-applications', 'human-llm-opinion-dynamics-moderation']
tags: ['counterspeech', 'abuse', 'robustness']
venue: "NAACL 2024"
tldr: "An investigation into the effectiveness of adversarial robustness training for abuse mitigation with counterspeech models."
---

# Multimodal Contextual Dialogue Breakdown Detection for Conversational AI Models

**Source**: [https://aclanthology.org/2024.naacl-industry.25/](https://aclanthology.org/2024.naacl-industry.25/)

**TLDR**: An investigation into the effectiveness of adversarial robustness training for abuse mitigation with counterspeech models.

## Abstract

AbstractDetecting dialogue breakdown in real time is critical for conversational AI systems, because it enables taking corrective action to successfully complete a task. In spoken dialog systems, this breakdown can be caused by a variety of unexpected situations including high levels of background noise, causing STT mistranscriptions, or unexpected user flows.In particular, industry settings like healthcare, require high precision and high flexibility to navigate differently based on the conversation history and dialogue states. This makes it both more challenging and more critical to accurately detect dialog breakdown. To accurately detect breakdown, we found it requires processing audio inputs along with downstream NLP model inferences on transcribed text in real time. In this paper, we introduce a Multimodal Contextual Dialogue Breakdown (MultConDB) model. This model significantly outperforms other known best models by achieving an F1 of 69.27.