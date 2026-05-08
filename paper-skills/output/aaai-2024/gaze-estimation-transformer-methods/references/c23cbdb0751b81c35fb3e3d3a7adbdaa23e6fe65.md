---
title: "Gaze Label Alignment: Alleviating Domain Shift for Gaze Estimation"
source: "https://www.semanticscholar.org/paper/c23cbdb0751b81c35fb3e3d3a7adbdaa23e6fe65"
categories: ['gaze-estimation-transformer-methods']
tags: ['gaze-estimation', 'domain-shift', 'label-alignment']
venue: "AAAI 2024"
tldr: "Addresses domain shift in gaze estimation by aligning label distributions rather than just data distributions."
---

# Gaze Label Alignment: Alleviating Domain Shift for Gaze Estimation

**Source**: [https://www.semanticscholar.org/paper/c23cbdb0751b81c35fb3e3d3a7adbdaa23e6fe65](https://www.semanticscholar.org/paper/c23cbdb0751b81c35fb3e3d3a7adbdaa23e6fe65)

**TLDR**: Addresses domain shift in gaze estimation by aligning label distributions rather than just data distributions.

## Abstract

Gaze estimation methods encounter significant performance deterioration when being evaluated across different domains, because of the domain gap between the testing and training data. Existing methods try to solve this issue by reducing the deviation of data distribution, however, they ignore the existence of label deviation in the data due to the acquisition mechanism of the gaze label and the individual physiological differences. In this paper, we first point out that the influence brought by the label deviation cannot be ignored, and propose a gaze label alignment algorithm (GLA) to eliminate the label distribution deviation. Specifically, we first train the feature extractor on all domains to get domain invariant features, and then select an anchor domain to train the gaze regressor. We predict the gaze label on remaining domains and use a mapping function to align the labels. Finally, these aligned labels can be used to train gaze estimation models. Therefore, our method can be combined with any existing method. Experimental results show that our GLA method can effectively alleviate the label distribution shift, and SOTA gaze estimation methods can be further improved obviously.