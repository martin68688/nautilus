---
title: "VPDETR: End-to-End Vanishing Point DEtection TRansformers"
source: "https://www.semanticscholar.org/paper/8a11e0d0d1acdffc37d8eb1b33ca2b64bc9e851f"
categories: ['vision-transformer-semi-supervised-learning']
tags: ['vanishing-point', 'detr', 'transformers']
venue: "AAAI 2024"
tldr: "Proposes an end-to-end transformer-based framework for vanishing point detection."
---

# VPDETR: End-to-End Vanishing Point DEtection TRansformers

**Source**: [https://www.semanticscholar.org/paper/8a11e0d0d1acdffc37d8eb1b33ca2b64bc9e851f](https://www.semanticscholar.org/paper/8a11e0d0d1acdffc37d8eb1b33ca2b64bc9e851f)

**TLDR**: Proposes an end-to-end transformer-based framework for vanishing point detection.

## Abstract

In the field of vanishing point detection, previous works commonly relied on extracting and clustering straight lines or classifying candidate points as vanishing points. This paper proposes a novel end-to-end framework, called VPDETR (Vanishing Point DEtection TRansformer), that views vanishing point detection as a set prediction problem, applicable to both Manhattan and non-Manhattan world datasets. By using the positional embedding of anchor points as queries in Transformer decoders and dynamically updating them layer by layer, our method is able to directly input images and output their vanishing points without the need for explicit straight line extraction and candidate points sampling. Additionally, we introduce an orthogonal loss and a cross-prediction loss to improve accuracy on the Manhattan world datasets. Experimental results demonstrate that VPDETR achieves competitive performance compared to state-of-the-art methods, without requiring post-processing.