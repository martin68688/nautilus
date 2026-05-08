---
title: "F-MALLOC: Feed-forward Memory Allocation for Continual Learning in Neural Machine Translation"
source: "https://aclanthology.org/2024.naacl-long.398/"
categories: ['continual-learning-memory-transfer-llms']
tags: ['continual-learning', 'neural-machine-translation', 'memory-allocation']
venue: "NAACL 2024"
tldr: "Introduces F-MALLOC, a feed-forward memory allocation method for continual learning in neural machine translation."
---

# F-MALLOC: Feed-forward Memory Allocation for Continual Learning in Neural Machine Translation

**Source**: [https://aclanthology.org/2024.naacl-long.398/](https://aclanthology.org/2024.naacl-long.398/)

**TLDR**: Introduces F-MALLOC, a feed-forward memory allocation method for continual learning in neural machine translation.

## Abstract

AbstractIn the evolving landscape of Neural Machine Translation (NMT), the pretrain-then-finetune paradigm has yielded impressive results. However, the persistent challenge of Catastrophic Forgetting (CF) remains a hurdle. While previous work has introduced Continual Learning (CL) methods to address CF, these approaches grapple with the delicate balance between avoiding forgetting and maintaining system extensibility. To address this, we propose a CL method, named F-MALLOC (Feed-forward Memory ALLOCation). F-MALLOC is inspired by recent insights highlighting that feed-forward layers emulate neural memories and encapsulate crucial translation knowledge. It decomposes feed-forward layers into discrete memory cells and allocates these memories to different tasks. By learning to allocate and safeguard these memories, our method effectively alleviates CF while ensuring robust extendability. Besides, we propose a comprehensive assessment protocol for multi-stage CL of NMT systems. Experiments conducted following this new protocol showcase the superior performance of F-MALLOC, evidenced by higher BLEU scores and almost zero forgetting.