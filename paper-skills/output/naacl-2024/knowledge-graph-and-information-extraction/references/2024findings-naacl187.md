---
title: "UNO-DST: Leveraging Unlabelled Data in Zero-Shot Dialogue State Tracking"
source: "https://aclanthology.org/2024.findings-naacl.187/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['dialogue-state-tracking', 'zero-shot', 'unlabeled-data']
venue: "NAACL 2024"
tldr: "Improves zero-shot DST by leveraging unlabeled target domain data through joint and self-training methods."
---

# UNO-DST: Leveraging Unlabelled Data in Zero-Shot Dialogue State Tracking

**Source**: [https://aclanthology.org/2024.findings-naacl.187/](https://aclanthology.org/2024.findings-naacl.187/)

**TLDR**: Improves zero-shot DST by leveraging unlabeled target domain data through joint and self-training methods.

## Abstract

AbstractPrevious zero-shot dialogue state tracking (DST) methods only apply transfer learning, but ignore unlabelled data in the target domain.We transform zero-shot DST into few-shot DST by utilising such unlabelled data via joint and self-training methods. Our method incorporates auxiliary tasks that generate slot types as inverse prompts for main tasks, creating slot values during joint training. Cycle consistency between these two tasks enables the generation and selection of quality samples in unknown target domains for subsequent fine-tuning. This approach also facilitates automatic label creation, thereby optimizing the training and fine-tuning of DST models. We demonstrate this method’s effectiveness on general language models in zero-shot scenarios, improving average joint goal accuracy by 8% across all domains in MultiWOZ.