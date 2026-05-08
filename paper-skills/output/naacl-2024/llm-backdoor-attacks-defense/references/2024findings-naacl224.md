---
title: "Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking"
source: "https://aclanthology.org/2024.findings-naacl.224/"
categories: ['llm-backdoor-attacks-defense', 'llm-alignment-safety-detoxification']
tags: ['jailbreak-attacks', 'logical-overload', 'safety-alignment', 'adversarial']
venue: "NAACL 2024"
tldr: "Introduces a novel jailbreak attack that overloads an LLM's logical processing to bypass safety alignments and elicit harmful responses."
---

# Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking

**Source**: [https://aclanthology.org/2024.findings-naacl.224/](https://aclanthology.org/2024.findings-naacl.224/)

**TLDR**: Introduces a novel jailbreak attack that overloads an LLM's logical processing to bypass safety alignments and elicit harmful responses.

## Abstract

AbstractWhile large language models (LLMs) have demonstrated increasing power, they have also called upon studies on their vulnerabilities. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of 1) multilingual cognitive overload, 2) veiled expression, and 3) effect-to- cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.