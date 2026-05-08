---
title: "The Role of n-gram Smoothing in the Age of Neural Networks"
source: "https://aclanthology.org/2024.naacl-long.382/"
categories: ['contrastive-and-generative-representation-learning', 'language-model-evaluation-benchmarks']
tags: ['language-model', 'n-gram', 'smoothing']
venue: "NAACL 2024"
tldr: "Re-examines the role of n-gram smoothing techniques in the context of modern neural language models."
---

# The Role of n-gram Smoothing in the Age of Neural Networks

**Source**: [https://aclanthology.org/2024.naacl-long.382/](https://aclanthology.org/2024.naacl-long.382/)

**TLDR**: Re-examines the role of n-gram smoothing techniques in the context of modern neural language models.

## Abstract

AbstractFor nearly three decades, language models derived from the n-gram assumption held the state of the art on the task. The key to their success lay in the application of various smoothing techniques that served to combat overfitting. However, when neural language models toppled n-gram models as the best performers, n-gram smoothing techniques became less relevant. Indeed, it would hardly be an understatement to suggest that the line of inquiry into n-gram smoothing techniques became dormant. This paper re-opens the role classical n-gram smoothing techniques may play in the age of neural language models. First, we draw a formal equivalence between label smoothing, a popular regularization technique for neural language models, and add-𝜆 smoothing. Second, we derive a generalized framework for converting any n-gram smoothing technique into a regularizer compatible with neural language models. Our empirical results find that our novel regularizers are comparable to and, indeed, sometimes outperform label smoothing on language modeling and machine translation.