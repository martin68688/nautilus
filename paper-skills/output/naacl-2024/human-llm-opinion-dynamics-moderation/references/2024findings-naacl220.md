---
title: "Z-GMOT: Zero-shot Generic Multiple Object Tracking"
source: "https://aclanthology.org/2024.findings-naacl.220/"
categories: ['human-llm-opinion-dynamics-moderation']
tags: ['agent-based-modeling', 'opinion-dynamics', 'simulation']
venue: "NAACL 2024"
tldr: "Proposes using networks of LLM-based agents to simulate human opinion dynamics, addressing the over-simplification of traditional models."
---

# Z-GMOT: Zero-shot Generic Multiple Object Tracking

**Source**: [https://aclanthology.org/2024.findings-naacl.220/](https://aclanthology.org/2024.findings-naacl.220/)

**TLDR**: Proposes using networks of LLM-based agents to simulate human opinion dynamics, addressing the over-simplification of traditional models.

## Abstract

AbstractDespite recent significant progress, Multi-Object Tracking (MOT) faces limitations such as reliance on prior knowledge and predefined categories and struggles with unseen objects. To address these issues, Generic Multiple Object Tracking (GMOT) has emerged as an alternative approach, requiring less prior information. However, current GMOT methods often rely on initial bounding boxes and struggle to handle variations in factors such as viewpoint, lighting, occlusion, and scale, among others. Our contributions commence with the introduction of the Referring GMOT dataset a collection of videos, each accompanied by detailed textual descriptions of their attributes. Subsequently, we propose Z-GMOT, a cutting-edge tracking solution capable of tracking objects from never-seen categories without the need of initial bounding boxes or predefined categories. Within our Z-GMOT framework, we introduce two novel components: (i) iGLIP, an improved Grounded language-image pretraining, for accurately detecting unseen objects with specific characteristics. (ii) MA-SORT, a novel object association approach that adeptly integrates motion and appearance-based matching strategies to tackle the complex task of tracking objects with high similarity. Our contributions are benchmarked through extensive experiments conducted on the Referring GMOT dataset for GMOT task. Additionally, to assess the generalizability of the proposed Z-GMOT, we conduct ablation studies on the DanceTrack and MOT20 datasets for the MOT task. Our dataset, code, and models are released at: https://fsoft-aic.github.io/Z-GMOT