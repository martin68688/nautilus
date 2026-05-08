---
title: "Weakly-Supervised Mirror Detection via Scribble Annotations"
source: "https://www.semanticscholar.org/paper/f3f1cdbfac9dee8facc629c13bd9f3eaed924de5"
categories: ['vision-transformer-semi-supervised-learning', 'lime-image-sampling-explanations']
tags: ['mirror-detection', 'weak-supervision', 'scribble-annotation']
venue: "AAAI 2024"
tldr: "Proposes a weakly-supervised mirror detection method using scribble annotations."
---

# Weakly-Supervised Mirror Detection via Scribble Annotations

**Source**: [https://www.semanticscholar.org/paper/f3f1cdbfac9dee8facc629c13bd9f3eaed924de5](https://www.semanticscholar.org/paper/f3f1cdbfac9dee8facc629c13bd9f3eaed924de5)

**TLDR**: Proposes a weakly-supervised mirror detection method using scribble annotations.

## Abstract

Mirror detection is of great significance for avoiding false recognition of reflected objects in computer vision tasks. Existing mirror detection frameworks usually follow a supervised setting, which relies heavily on high quality labels and suffers from poor generalization. To resolve this, we instead propose the first weakly-supervised mirror detection framework and also provide the first scribble-based mirror dataset. Specifically, we relabel 10,158 images, most of which have a labeled pixel ratio of less than 0.01 and take only about 8 seconds to label. Considering that the mirror regions usually show great scale variation, and also irregular and occluded, thus leading to issues of incomplete or over detection, we propose a local-global feature enhancement (LGFE) module to fully capture the context and details. Moreover, it is difficult to obtain basic mirror structure using scribble annotation, and the distinction between foreground (mirror) and background (non-mirror) features is not emphasized caused by mirror reflections. Therefore, we propose a foreground-aware mask attention (FAMA), integrating mirror edges and semantic features to complete mirror regions and suppressing the influence of backgrounds. Finally, to improve the robustness of the network, we propose a prototype contrast loss (PCL) to learn more general foreground features across images. Extensive experiments show that our network outperforms relevant state-of-the-art weakly supervised methods, and even some fully supervised methods. The dataset and codes are available at https://github.com/winter-flow/WSMD.