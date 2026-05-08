---
title: "MuMath: Multi-perspective Data Augmentation for Mathematical Reasoning in Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.185/"
categories: ['efficient-transformer-optimization-techniques', 'llm-education-assessment-augmentation']
tags: ['data-augmentation', 'mathematical-reasoning', 'tool-use']
venue: "NAACL 2024"
tldr: "Proposes multi-perspective data augmentation to improve the calculation process transparency and reasoning of tool-use LLMs for math."
---

# MuMath: Multi-perspective Data Augmentation for Mathematical Reasoning in Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.185/](https://aclanthology.org/2024.findings-naacl.185/)

**TLDR**: Proposes multi-perspective data augmentation to improve the calculation process transparency and reasoning of tool-use LLMs for math.

## Abstract

AbstractRecently, the tool-use Large Language Models (LLMs) that integrate with external Python interpreters have significantly enhanced mathematical reasoning capabilities for open-source LLMs. However, these models fall short in demonstrating the calculation process, which compromises user-friendliness and understanding of problem-solving steps. Conversely, while tool-free methods offer a clear display of the problem-solving process, their accuracy leaves room for improvement.These tool-free methods typically employ a somewhat narrow range of augmentation techniques such as rephrasing and difficulty enhancement to boost performance. In response to this issue, we have amalgamated and further refined these strengths while broadening the scope of augmentation methods to construct a **mu**lti-perspective augmentation dataset for **math**ematics—termed **MuMath** (𝜇-Math) Dataset.Subsequently, we finetune LLaMA-2 on the MuMath dataset to derive the MuMath model. Our experiments indicate that our MuMath-70B model achieves new state-of-the-art performance among tool-free methods—achieving 88.3% on GSM8K and 34.5% on MATH .We release the MuMath dataset along with its corresponding models and code for public use.