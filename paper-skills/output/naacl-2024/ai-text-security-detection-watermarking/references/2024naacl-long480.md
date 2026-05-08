---
title: "Keep it Private: Unsupervised Privatization of Online Text"
source: "https://aclanthology.org/2024.naacl-long.480/"
categories: ['metaphor-analysis-political-framing', 'ai-text-security-detection-watermarking']
tags: ['privacy', 'anonymization', 'style-transfer']
venue: "NAACL 2024"
tldr: "Proposes an unsupervised method to privatize online text for authorship obfuscation."
---

# Keep it Private: Unsupervised Privatization of Online Text

**Source**: [https://aclanthology.org/2024.naacl-long.480/](https://aclanthology.org/2024.naacl-long.480/)

**TLDR**: Proposes an unsupervised method to privatize online text for authorship obfuscation.

## Abstract

AbstractAuthorship obfuscation techniques hold the promise of helping people protect their privacy in online communications by automatically rewriting text to hide the identity of the original author. However, obfuscation has been evaluated in narrow settings in the NLP literature and has primarily been addressed with superficial edit operations that can lead to unnatural outputs. In this work, we introduce an automatic text privatization framework that fine-tunes a large language model via reinforcement learning to produce rewrites that balance soundness, sense, and privacy. We evaluate it extensively on a large-scale test set of English Reddit posts by 68k authors composed of short-medium length texts. We study how the performance changes among evaluative conditions including authorial profile length and authorship detection strategy. Our method maintains high text quality according to both automated metrics and human evaluation, and successfully evades several automated authorship attacks.