---
title: "LatticeGen: Hiding Generated Text in a Lattice for Privacy-Aware Large Language Model Generation on Cloud"
source: "https://aclanthology.org/2024.findings-naacl.171/"
categories: ['llm-backdoor-attacks-defense', 'continual-learning-memory-transfer-llms']
tags: ['privacy', 'secure-generation', 'watermarking']
venue: "NAACL 2024"
tldr: "LatticeGen is a method for privacy-aware LLM generation on cloud by hiding generated text in a lattice structure controlled by the user."
---

# LatticeGen: Hiding Generated Text in a Lattice for Privacy-Aware Large Language Model Generation on Cloud

**Source**: [https://aclanthology.org/2024.findings-naacl.171/](https://aclanthology.org/2024.findings-naacl.171/)

**TLDR**: LatticeGen is a method for privacy-aware LLM generation on cloud by hiding generated text in a lattice structure controlled by the user.

## Abstract

AbstractIn the current user-server interaction paradigm of prompted generation with large language models (LLMs) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text private to themselves. For privacy-aware text generation on cloud, we propose LatticeGen, a cooperative protocol in which the server still handles most of the computation while the client controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the client and hidden in a noised lattice. Only the client knows which tokens are the true ones. Considering potential attacks from a hypothetically malicious server and how the client can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).