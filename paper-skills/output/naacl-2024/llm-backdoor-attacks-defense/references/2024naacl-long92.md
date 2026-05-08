---
title: "SELF-GUARD: Empower the LLM to Safeguard Itself"
source: "https://aclanthology.org/2024.naacl-long.92/"
categories: ['llm-backdoor-attacks-defense', 'llm-alignment-safety-detoxification']
tags: ['jailbreak-defense', 'self-safeguard', 'llm-safety']
venue: "NAACL 2024"
tldr: "Empowers LLMs to self-safeguard against jailbreak attacks by integrating a safety module that guides response generation."
---

# SELF-GUARD: Empower the LLM to Safeguard Itself

**Source**: [https://aclanthology.org/2024.naacl-long.92/](https://aclanthology.org/2024.naacl-long.92/)

**TLDR**: Empowers LLMs to self-safeguard against jailbreak attacks by integrating a safety module that guides response generation.

## Abstract

AbstractWith the increasing risk posed by jailbreak attacks, recent studies have investigated various methods to improve the safety of large language models (LLMs), mainly falling into two strategies: safety training and safeguards. Safety training involves fine-tuning the LLM with adversarial samples, which activate the LLM’s capabilities against jailbreak. However, it is not always effective in countering new attacks and often leads to potential performance degradation. Safeguards, on the other hand, are methods using additional models to filter harmful content from the LLM’s response. Nevertheless, they can only reduce a limited amount of harmful output and introduce extra computational costs. Given the distinct strengths and weaknesses of both, we combine them to balance out their flaws and propose a more effective method called Self-Guard.Specifically, we train the LLM to review its responses for any harmful content and append a [harmful] or [harmless] tag to the end of the response. In this way, Self-Guard possesses the advantages of safety training, leveraging the powerful capabilities of the LLMs themselves to detect harmfulness. Besides that, it gains flexibility like safeguards, making the safety check target the output side, which makes the system less vulnerable to attack updates. Experimental results indicate that our Self-Guard can effectively defend against jailbreak attacks and will not cause LLMs’ performance degradation.