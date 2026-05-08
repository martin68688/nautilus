---
title: "Evaluating Step-by-Step Reasoning through Symbolic Verification"
source: "https://aclanthology.org/2024.findings-naacl.188/"
categories: ['llm-alignment-safety-detoxification', 'logical-reasoning-in-neural-models']
tags: ['reasoning', 'symbolic-verification', 'chain-of-thought']
venue: "NAACL 2024"
tldr: "Evaluates the step-by-step reasoning of language models by verifying their logical chains against symbolic programs."
---

# Evaluating Step-by-Step Reasoning through Symbolic Verification

**Source**: [https://aclanthology.org/2024.findings-naacl.188/](https://aclanthology.org/2024.findings-naacl.188/)

**TLDR**: Evaluates the step-by-step reasoning of language models by verifying their logical chains against symbolic programs.

## Abstract

AbstractPre-trained language models (LMs) have shown remarkable reasoning performance using explanations or chain-of-thoughts (CoT)) for in-context learning. On the other hand, these reasoning tasks are usually presumed to be more approachable for symbolic programming. To understand the mechanism of reasoning of LMs, we curate synthetic datasets containing equivalent (natural, symbolic) data pairs, where symbolic examples contain first-order logic rules and predicates from non-parametric knowledge bases (KBs), supporting automated verification of intermediate reasoning results. Then we revisit neuro-symbolic approaches and propose to learn from demonstrations containing logic rules and corresponding examples to iteratively reason over KBs, recovering Prolog’s backward chaining algorithm and supporting automated verification of LMs’ outputs. Comprehensive experiments are included to systematically compare LMLP with CoT in deductive reasoning settings, showing that LMLP enjoys more than 25% higher accuracy than CoT on length generalization benchmarks even with smaller model sizes.