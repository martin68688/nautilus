---
title: "Advancing Beyond Identification: Multi-bit Watermark for Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.224/"
categories: ['ai-text-security-detection-watermarking']
tags: ['watermarking', 'multi-bit', 'security']
venue: "NAACL 2024"
tldr: "Proposes a multi-bit watermark for LLMs to trace adversary users, advancing beyond mere identification of machine-generated text."
---

# Advancing Beyond Identification: Multi-bit Watermark for Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.224/](https://aclanthology.org/2024.naacl-long.224/)

**TLDR**: Proposes a multi-bit watermark for LLMs to trace adversary users, advancing beyond mere identification of machine-generated text.

## Abstract

AbstractWe show the viability of tackling misuses of large language models beyond the identification of machine-generated text. While existing zero-bit watermark methods focus on detection only, some malicious misuses demand tracing the adversary user for counteracting them. To address this, we propose Multi-bit Watermark via Position Allocation, embedding traceable multi-bit information during language model generation. Through allocating tokens onto different parts of the messages, we embed longer messages in high corruption settings without added latency. By independently embedding sub-units of messages, the proposed method outperforms the existing works in terms of robustness and latency. Leveraging the benefits of zero-bit watermarking, our method enables robust extraction of the watermark without any model access, embedding and extraction of long messages (≥ 32-bit) without finetuning, and maintaining text quality, while allowing zero-bit detection all at the same time.