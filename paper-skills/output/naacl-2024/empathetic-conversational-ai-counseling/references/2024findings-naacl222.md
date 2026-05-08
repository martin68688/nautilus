---
title: "PRODIGy: a PROfile-based DIalogue Generation dataset"
source: "https://aclanthology.org/2024.findings-naacl.222/"
categories: ['legal-ai-nlp-applications', 'empathetic-conversational-ai-counseling']
tags: ['dialogue-generation', 'profile-based', 'dataset']
venue: "NAACL 2024"
tldr: "Introduces PRODIGy, a dataset for training profile-based dialogue agents with complex, implicit profile representations."
---

# PRODIGy: a PROfile-based DIalogue Generation dataset

**Source**: [https://aclanthology.org/2024.findings-naacl.222/](https://aclanthology.org/2024.findings-naacl.222/)

**TLDR**: Introduces PRODIGy, a dataset for training profile-based dialogue agents with complex, implicit profile representations.

## Abstract

AbstractProviding dialogue agents with a profile representation can improve their consistency and coherence, leading to better conversations. However, current profile-based dialogue datasets for training such agents contain either explicit profile representations that are simple and dialogue-specific, or implicit representations that are difficult to collect. In this work, we introduce the PRODIGy (PROfile-based DIalogue Generation) dataset, which brings diverse representations together, providing a more comprehensive profile dimension set for each speaker. This resource comprises more than 20k dialogues, sourced from movie scripts, aligned with speaker representations such as communication style, biography, personality and gender. Initial experiments with diverse baselines show that providing generative language models with these aspects of a profile, both separately and jointly, enhances models’ performance. This improvement holds true in both in-domain and cross-domain settings, for both fine-tuned and instruction-based LLMs.