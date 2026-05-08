---
title: "A Robust Semantics-based Watermark for Large Language Model against Paraphrasing"
source: "https://aclanthology.org/2024.findings-naacl.40/"
categories: ['ai-text-security-detection-watermarking']
tags: ['watermarking', 'security', 'paraphrasing', 'semantics']
venue: "NAACL 2024"
tldr: "Presents a robust semantics-based watermark for LLM-generated text to detect misuse, resilient to paraphrasing."
---

# A Robust Semantics-based Watermark for Large Language Model against Paraphrasing

**Source**: [https://aclanthology.org/2024.findings-naacl.40/](https://aclanthology.org/2024.findings-naacl.40/)

**TLDR**: Presents a robust semantics-based watermark for LLM-generated text to detect misuse, resilient to paraphrasing.

## Abstract

AbstractLarge language models (LLMs) have show their remarkable ability in various natural language tasks. However, there are concerns that LLMs are possible to be used improperly or even illegally. To prevent the malicious usage of LLMs, detecting LLM-generated text becomes crucial in the deployment of LLM applications. Watermarking is an effective strategy to detect the LLM-generated content by encoding a pre-defined secret watermark to facilitate the detection process. However, the majority of existing watermark methods leverage the simple hashes of precedent tokens to partition vocabulary. Such watermarks can be easily eliminated by paraphrase and, correspondingly, the detection effectiveness will be greatly compromised. Thus, to enhance the robustness against paraphrase, we propose a semantics-based watermark framework, SemaMark. It leverages the semantics as an alternative to simple hashes of tokens since the semantic meaning of the sentences will be likely preserved under paraphrase and the watermark can remain robust. Comprehensive experiments are conducted to demonstrate the effectiveness and robustness of SemaMark under different paraphrases.