---
title: "UniverSLU: Universal Spoken Language Understanding for Diverse Tasks with Natural Language Instructions"
source: "https://aclanthology.org/2024.naacl-long.151/"
categories: ['llm-evaluation-summarization-argument-extraction', 'speech-language-processing-multilingual']
tags: ['spoken-language-understanding', 'instruction-tuning', 'multitask']
venue: "NAACL 2024"
tldr: "Builds a single universal model for diverse spoken language understanding tasks using natural language instructions."
---

# UniverSLU: Universal Spoken Language Understanding for Diverse Tasks with Natural Language Instructions

**Source**: [https://aclanthology.org/2024.naacl-long.151/](https://aclanthology.org/2024.naacl-long.151/)

**TLDR**: Builds a single universal model for diverse spoken language understanding tasks using natural language instructions.

## Abstract

AbstractRecent studies leverage large language models with multi-tasking capabilities, using natural language prompts to guide the model’s behavior and surpassing performance of task-specific models. Motivated by this, we ask: can we build a single model that jointly performs various spoken language understanding (SLU) tasks? We start by adapting a pre-trained automatic speech recognition model to additional tasks using single-token task specifiers. We enhance this approach through instruction tuning, i.e., finetuning by describing the task using natural language instructions followed by the list of label options. Our approach can generalize to new task descriptions for the seen tasks during inference, thereby enhancing its user-friendliness. We demonstrate the efficacy of our single multi-task learning model “UniverSLU” for 12 speech classification and sequence generation task types spanning 17 datasets and 9 languages. On most tasks, UniverSLU achieves competitive performance and often even surpasses task-specific models. Additionally, we assess the zero-shot capabilities, finding that the model generalizes to new datasets and languages for seen task types.