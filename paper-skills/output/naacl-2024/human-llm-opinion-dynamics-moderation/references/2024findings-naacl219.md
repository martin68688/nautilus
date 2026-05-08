---
title: "Don’t be a Fool: Pooling Strategies in Offensive Language Detection from User-Intended Adversarial Attacks"
source: "https://aclanthology.org/2024.findings-naacl.219/"
categories: ['human-llm-opinion-dynamics-moderation', 'social-bias-mitigation-in-language-models']
tags: ['offensive-language-detection', 'adversarial-attacks', 'pooling-strategies']
venue: "NAACL 2024"
tldr: "Proposes pooling strategies to improve offensive language detection robustness against user-intended adversarial text attacks."
---

# Don’t be a Fool: Pooling Strategies in Offensive Language Detection from User-Intended Adversarial Attacks

**Source**: [https://aclanthology.org/2024.findings-naacl.219/](https://aclanthology.org/2024.findings-naacl.219/)

**TLDR**: Proposes pooling strategies to improve offensive language detection robustness against user-intended adversarial text attacks.

## Abstract

AbstractOffensive language detection is an important task for filtering out abusive expressions and improving online user experiences. However, malicious users often attempt to avoid filtering systems through the involvement of textual noises. In this paper, we propose these evasions as user-intended adversarial attacks that insert special symbols or leverage the distinctive features of the Korean language. Furthermore, we introduce simple yet effective pooling strategies in a layer-wise manner to defend against the proposed attacks, focusing on the preceding layers not just the last layer to capture both offensiveness and token embeddings. We demonstrate that these pooling strategies are more robust to performance degradation even when the attack rate is increased, without directly training of such patterns. Notably, we found that models pre-trained on clean texts could achieve a comparable performance in detecting attacked offensive language, to models pre-trained on noisy texts by employing these pooling strategies.