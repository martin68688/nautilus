---
title: "MedCycle: Unpaired Medical Report Generation via Cycle-Consistency"
source: "https://aclanthology.org/2024.findings-naacl.125/"
categories: ['clinical-nlp-biomedical-applications']
tags: ['medical-report-generation', 'unpaired', 'cycle-consistency', 'multimodal']
venue: "NAACL 2024"
tldr: "Generating medical reports from X-ray images using cycle-consistency in an unpaired training scenario."
---

# MedCycle: Unpaired Medical Report Generation via Cycle-Consistency

**Source**: [https://aclanthology.org/2024.findings-naacl.125/](https://aclanthology.org/2024.findings-naacl.125/)

**TLDR**: Generating medical reports from X-ray images using cycle-consistency in an unpaired training scenario.

## Abstract

AbstractGenerating medical reports for X-ray images presents a significant challenge, particularly in unpaired scenarios where access to paired image-report data for training is unavailable. Previous works have typically learned a joint embedding space for images and reports, necessitating a specific labeling schema for both. We introduce an innovative approach that eliminates the need for consistent labeling schemas, thereby enhancing data accessibility and enabling the use of incompatible datasets. This approach is based on cycle-consistent mapping functions that transform image embeddings into report embeddings, coupled with report auto encoding for medical report generation. Our model and objectives consider intricate local details and the overarching semantic context within images and reports. This approach facilitates the learning of effective mapping functions, resulting in the generation of coherent reports. It outperforms state-of-the-art results in unpaired chest X-ray report generation, demonstrating improvements in both language and clinical metrics.