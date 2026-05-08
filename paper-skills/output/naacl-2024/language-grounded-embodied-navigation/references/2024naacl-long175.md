---
title: "Which One? Leveraging Context Between Objects and Multiple Views for Language Grounding"
source: "https://aclanthology.org/2024.naacl-long.175/"
categories: ['language-grounded-embodied-navigation', 'efficient-transformer-optimization-techniques']
tags: ['embodied-ai', 'language-grounding', '3d-scenes']
venue: "NAACL 2024"
tldr: "Improves language grounding in 3D environments by using context between objects and multiple camera views."
---

# Which One? Leveraging Context Between Objects and Multiple Views for Language Grounding

**Source**: [https://aclanthology.org/2024.naacl-long.175/](https://aclanthology.org/2024.naacl-long.175/)

**TLDR**: Improves language grounding in 3D environments by using context between objects and multiple camera views.

## Abstract

AbstractWhen connecting objects and their language referents in an embodied 3D environment, it is important to note that: (1) an object can be better characterized by leveraging comparative information between itself and other objects, and (2) an object’s appearance can vary with camera position. As such, we present the Multi-view Approach to Grounding in Context (MAGiC) model, which selects an object referent based on language that distinguishes between two similar objects. By pragmatically reasoning over both objects and across multiple views of those objects, MAGiC improves over the state-of-the-art model on the SNARE object reference task with a relative error reduction of 12.9% (representing an absolute improvement of 2.7%). Ablation studies show that reasoning jointly over object referent candidates and multiple views of each object both contribute to improved accuracy. Code: https://github.com/rcorona/magic_snare/