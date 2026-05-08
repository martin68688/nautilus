---
title: "LLM-as-a-Coauthor: Can Mixed Human-Written and Machine-Generated Text Be Detected?"
source: "https://aclanthology.org/2024.findings-naacl.29/"
categories: ['llm-backdoor-attacks-defense', 'knowledge-graph-and-information-extraction', 'large-language-model-evaluation-augmentation']
tags: ['machine-generated-text', 'detection', 'human-ai-collaboration', 'integrity']
venue: "NAACL 2024"
tldr: "Investigates the detection of mixed human-written and machine-generated text, highlighting the challenges posed by LLM-assisted authorship."
---

# LLM-as-a-Coauthor: Can Mixed Human-Written and Machine-Generated Text Be Detected?

**Source**: [https://aclanthology.org/2024.findings-naacl.29/](https://aclanthology.org/2024.findings-naacl.29/)

**TLDR**: Investigates the detection of mixed human-written and machine-generated text, highlighting the challenges posed by LLM-assisted authorship.

## Abstract

AbstractWith the rapid development and widespread application of Large Language Models (LLMs), the use of Machine-Generated Text (MGT) has become increasingly common, bringing with it potential risks, especially in terms of quality and integrity in fields like news, education, and science. Current research mainly focuses on purely MGT detection, without adequately addressing mixed scenarios including AI-revised Human-Written Text (HWT) or human-revised MGT. To tackle this challenge, we define mixtext, a form of mixed text involving both AI and human-generated content. Then we introduce MixSet, the first dataset dedicated to studying these mixtext scenarios. Leveraging MixSet, we executed comprehensive experiments to assess the efficacy of prevalent MGT detectors in handling mixtext situations, evaluating their performance in terms of effectiveness, robustness, and generalization. Our findings reveal that existing detectors struggle to identify mixtext, particularly in dealing with subtle modifications and style adaptability. This research underscores the urgent need for more fine-grain detectors tailored for mixtext, offering valuable insights for future research. Code and Models are available at https://github.com/Dongping-Chen/MixSet.