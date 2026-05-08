---
title: "OSCaR: Object State Captioning and State Change Representation"
source: "https://aclanthology.org/2024.findings-naacl.226/"
categories: ['bpe-subword-parsing-evaluation', 'efficient-transformer-optimization-techniques']
tags: ['bpe', 'subword', 'parsing', 'evaluation', 'multilingual']
venue: "NAACL 2024"
tldr: "OSCaR benchmarks object state captioning and change representation for AI models."
---

# OSCaR: Object State Captioning and State Change Representation

**Source**: [https://aclanthology.org/2024.findings-naacl.226/](https://aclanthology.org/2024.findings-naacl.226/)

**TLDR**: OSCaR benchmarks object state captioning and change representation for AI models.

## Abstract

AbstractThe capability of intelligent models to extrapolate and comprehend changes in object states is a crucial yet demanding aspect of AI research, particularly through the lens of human interaction in real-world settings. This task involves describing complex visual environments, identifying active objects, and interpreting their changes as conveyed through language. Traditional methods, which isolate object captioning and state change detection, offer a limited view of dynamic environments. Moreover, relying on a small set of symbolic words to represent changes has restricted the expressiveness of language. To address these challenges, in this paper, we introduce the Object State Captioning and State Change Representation (OSCaR) dataset and benchmark. OSCaR consists of 14,084 annotated video segments with nearly 1,000 unique objects from various egocentric video collections. It sets a new testbed for evaluating Multimodal Large Language Models (MLLMs). Our experiments demonstrate that while MLLMs show some skill, they lack a full understanding of object state changes. The benchmark includes a fine-tuned model that, despite initial capabilities, requires significant improvements in accuracy and generalization ability for effective understanding of these changes. Our code and dataset are available at https://github.com/nguyennm1024/OSCaR.