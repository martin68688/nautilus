---
title: "Semantically-Prompted Language Models Improve Visual Descriptions"
source: "https://aclanthology.org/2024.findings-naacl.267/"
categories: ['knowledge-conflict-diagnostic-temporal-reasoning', 'text-guided-multimodal-generation']
tags: ['vision-language', 'image-description', 'semantic-prompting']
venue: "NAACL 2024"
tldr: "Uses semantically-prompted language models to generate more specific and granular visual descriptions."
---

# Semantically-Prompted Language Models Improve Visual Descriptions

**Source**: [https://aclanthology.org/2024.findings-naacl.267/](https://aclanthology.org/2024.findings-naacl.267/)

**TLDR**: Uses semantically-prompted language models to generate more specific and granular visual descriptions.

## Abstract

AbstractLanguage-vision models like CLIP have made significant strides in vision tasks, such as zero-shot image classification (ZSIC). However, generating specific and expressive visual descriptions remains challenging; descriptions produced by current methods are often ambiguous and lacking in granularity. To tackle these issues, we propose V-GLOSS: Visual Glosses, a novel method built upon two key ideas. The first is Semantic Prompting, which conditions a language model on structured semantic knowledge. The second is a new contrastive algorithm that elicits fine-grained distinctions between similar concepts. With both ideas, we demonstrate that V-GLOSS improves visual descriptions and achieves strong results in the zero-shot setting on general and fine-grained image-classification datasets, including ImageNet, STL-10, FGVC Aircraft, and Flowers 102. Moreover, these descriptive capabilities contribute to enhancing image-generation performance. Finally, we introduce a quality-tested silver dataset with descriptions generated with V-GLOSS for all ImageNet classes.