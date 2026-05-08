---
title: "A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily"
source: "https://aclanthology.org/2024.naacl-long.118/"
categories: ['bpe-subword-parsing-evaluation', 'continual-learning-memory-transfer-llms']
tags: ['multimodal', 'VQA', 'transformer']
venue: "NAACL 2024"
tldr: "A novel encoder-decoder transformer approach for text-VQA that processes multiple questions and their associated content to predict multiple answers."
---

# A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily

**Source**: [https://aclanthology.org/2024.naacl-long.118/](https://aclanthology.org/2024.naacl-long.118/)

**TLDR**: A novel encoder-decoder transformer approach for text-VQA that processes multiple questions and their associated content to predict multiple answers.

## Abstract

AbstractLarge Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as ‘jailbreaks’ can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.