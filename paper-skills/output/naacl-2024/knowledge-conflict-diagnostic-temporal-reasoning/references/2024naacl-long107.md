---
title: "MART: Improving LLM Safety with Multi-round Automatic Red-Teaming"
source: "https://aclanthology.org/2024.naacl-long.107/"
categories: ['knowledge-conflict-diagnostic-temporal-reasoning', 'llm-alignment-safety-detoxification']
tags: ['red-teaming', 'safety', 'multi-round', 'automated']
venue: "NAACL 2024"
tldr: "Proposes a multi-round automatic red-teaming method to improve LLM safety by iteratively generating and filtering adversarial prompts."
---

# MART: Improving LLM Safety with Multi-round Automatic Red-Teaming

**Source**: [https://aclanthology.org/2024.naacl-long.107/](https://aclanthology.org/2024.naacl-long.107/)

**TLDR**: Proposes a multi-round automatic red-teaming method to improve LLM safety by iteratively generating and filtering adversarial prompts.

## Abstract

AbstractRed-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses.While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them.In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM.Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning.On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.