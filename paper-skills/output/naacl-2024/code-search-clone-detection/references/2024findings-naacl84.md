---
title: "Learning Cross-Architecture Instruction Embeddings for Binary Code Analysis in Low-Resource Architectures"
source: "https://aclanthology.org/2024.findings-naacl.84/"
categories: ['code-search-clone-detection']
tags: ['binary-code', 'instruction-embeddings', 'cross-architecture']
venue: "NAACL 2024"
tldr: "Learns cross-architecture instruction embeddings for binary code analysis to improve performance on low-resource instruction set architectures."
---

# Learning Cross-Architecture Instruction Embeddings for Binary Code Analysis in Low-Resource Architectures

**Source**: [https://aclanthology.org/2024.findings-naacl.84/](https://aclanthology.org/2024.findings-naacl.84/)

**TLDR**: Learns cross-architecture instruction embeddings for binary code analysis to improve performance on low-resource instruction set architectures.

## Abstract

AbstractBinary code analysis is indispensable for a variety of software security tasks. Applying deep learning to binary code analysis has drawn great attention because of its notable performance. Today, source code is frequently compiled for various Instruction Set Architectures (ISAs). It is thus critical to expand binary analysis capabilities to multiple ISAs. Given a binary analysis task, the scale of available data on different ISAs varies. As a result, the rich datasets (e.g., malware) for certain ISAs, such as x86, lead to a disproportionate focus on these ISAs and a negligence of other ISAs, such as PowerPC, which suffer from the “data scarcity” problem. To address the problem, we propose to learn cross-architecture instruction embeddings (CAIE), where semantically-similar instructions, regardless of their ISAs, have close embeddings in a shared space. Consequently, we can transfer a model trained on a data-rich ISA to another ISA with less available data. We consider four ISAs (x86, ARM, MIPS, and PowerPC) and conduct both intrinsic and extrinsic evaluations (including malware detection and function similarity comparison). The results demonstrate the effectiveness of our approach to generate high-quality CAIE with good transferability.