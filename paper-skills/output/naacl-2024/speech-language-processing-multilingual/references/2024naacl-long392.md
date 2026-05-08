---
title: "Document Image Machine Translation with Dynamic Multi-pre-trained Models Assembling"
source: "https://aclanthology.org/2024.naacl-long.392/"
categories: ['speech-language-processing-multilingual']
tags: ['document-translation', 'multimodal', 'machine-translation', 'model-assembling']
venue: "NAACL 2024"
tldr: "This paper proposes a novel document image machine translation task and a dynamic multi-pre-trained model assembling method."
---

# Document Image Machine Translation with Dynamic Multi-pre-trained Models Assembling

**Source**: [https://aclanthology.org/2024.naacl-long.392/](https://aclanthology.org/2024.naacl-long.392/)

**TLDR**: This paper proposes a novel document image machine translation task and a dynamic multi-pre-trained model assembling method.

## Abstract

AbstractText image machine translation (TIMT) is a task that translates source texts embedded in the image to target translations. The existing TIMT task mainly focuses on text-line-level images. In this paper, we extend the current TIMT task and propose a novel task, **D**ocument **I**mage **M**achine **T**ranslation to **Markdown** (**DIMT2Markdown**), which aims to translate a source document image with long context and complex layout structure to markdown-formatted target translation.We also introduce a novel framework, **D**ocument **I**mage **M**achine **T**ranslation with **D**ynamic multi-pre-trained models **A**ssembling (**DIMTDA**).A dynamic model assembler is used to integrate multiple pre-trained models to enhance the model’s understanding of layout and translation capabilities.Moreover, we build a novel large-scale **Do**cument image machine **T**ranslation dataset of **A**rXiv articles in markdown format (**DoTA**), containing 126K image-translation pairs.Extensive experiments demonstrate the feasibility of end-to-end translation of rich-text document images and the effectiveness of DIMTDA.