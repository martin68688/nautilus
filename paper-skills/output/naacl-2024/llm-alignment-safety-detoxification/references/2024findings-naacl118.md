---
title: "Adversarial DPO: Harnessing Harmful Data for Reducing Toxicity with Minimal Impact on Coherence and Evasiveness in Dialogue Agents"
source: "https://aclanthology.org/2024.findings-naacl.118/"
categories: ['llm-alignment-safety-detoxification']
tags: ['alignment', 'detoxification', 'adversarial-training']
venue: "NAACL 2024"
tldr: "Proposes Adversarial DPO, a method using harmful data in DPO training to reduce LLM toxicity with minimal impact on coherence and evasiveness."
---

# Adversarial DPO: Harnessing Harmful Data for Reducing Toxicity with Minimal Impact on Coherence and Evasiveness in Dialogue Agents

**Source**: [https://aclanthology.org/2024.findings-naacl.118/](https://aclanthology.org/2024.findings-naacl.118/)

**TLDR**: Proposes Adversarial DPO, a method using harmful data in DPO training to reduce LLM toxicity with minimal impact on coherence and evasiveness.

## Abstract

AbstractRecent advancements in open-domain dialogue systems have been propelled by the emergence of high-quality large language models (LLMs) and various effective training methodologies. Nevertheless, the presence of toxicity within these models presents a significant challenge that can potentially diminish the user experience. In this study, we introduce an innovative training algorithm, an improvement upon direct preference optimization (DPO), called adversarial DPO (ADPO). The ADPO algorithm is designed to train models to assign higher probability distributions to preferred responses and lower distributions to unsafe responses, which are self-generated using the toxic control token. We demonstrate that ADPO enhances the model’s resilience against harmful conversations while minimizing performance degradation. Furthermore, we illustrate that ADPO offers a more stable training procedure compared to the traditional DPO. To the best of our knowledge, this is the first adaptation of the DPO algorithm that directly incorporates harmful data into the generative model, thereby reducing the need to artificially create safe dialogue data.