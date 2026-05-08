---
title: "LEEETs-Dial: Linguistic Entrainment in End-to-End Task-oriented Dialogue systems"
source: "https://aclanthology.org/2024.findings-naacl.46/"
categories: ['code-switch-entrainment-measurement']
tags: ['entrainment', 'dialogue-systems', 'task-oriented']
venue: "NAACL 2024"
tldr: "Introduces a method to incorporate linguistic entrainment into end-to-end task-oriented dialogue systems for a more natural user experience."
---

# LEEETs-Dial: Linguistic Entrainment in End-to-End Task-oriented Dialogue systems

**Source**: [https://aclanthology.org/2024.findings-naacl.46/](https://aclanthology.org/2024.findings-naacl.46/)

**TLDR**: Introduces a method to incorporate linguistic entrainment into end-to-end task-oriented dialogue systems for a more natural user experience.

## Abstract

AbstractLinguistic entrainment, or alignment, represents a phenomenon where linguistic patterns employed by conversational participants converge to one another. While entrainment has been shown to produce a more natural user experience, most dialogue systems do not have any provisions for it. In this work, we introduce methods for achieving dialogue entrainment in a GPT-2-based end-to-end task-oriented dialogue system through the utilization of shared vocabulary. We experiment with training instance weighting, entrainment-specific loss, and additional conditioning to generate responses that align with the user. We demonstrate that all three approaches produce significantly better entrainment than the base, non-entrainment-optimized model, as confirmed by both automated and manual evaluation metrics.