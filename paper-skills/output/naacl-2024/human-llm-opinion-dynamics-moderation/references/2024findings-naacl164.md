---
title: "Tailoring Vaccine Messaging with Common-Ground Opinions"
source: "https://aclanthology.org/2024.findings-naacl.164/"
categories: ['contrastive-and-generative-representation-learning', 'human-llm-opinion-dynamics-moderation']
tags: ['personalization', 'vaccine-messaging', 'common-ground']
venue: "NAACL 2024"
tldr: "A method tailors vaccine messaging by establishing common-ground opinions to address concerns and misinformation."
---

# Tailoring Vaccine Messaging with Common-Ground Opinions

**Source**: [https://aclanthology.org/2024.findings-naacl.164/](https://aclanthology.org/2024.findings-naacl.164/)

**TLDR**: A method tailors vaccine messaging by establishing common-ground opinions to address concerns and misinformation.

## Abstract

AbstractOne way to personalize chatbot interactions is by establishing common ground with the intended reader. A domain where establishing mutual understanding could be particularly impactful is vaccine concerns and misinformation. Vaccine interventions are forms of messaging which aim to answer concerns expressed about vaccination. Tailoring responses in this domain is difficult, since opinions often have seemingly little ideological overlap. We define the task of tailoring vaccine interventions to a Common-Ground Opinion (CGO). Tailoring responses to a CGO involves meaningfully improving the answer by relating it to an opinion or belief the reader holds. In this paper we introduce Tailor-CGO, a dataset for evaluating how well responses are tailored to provided CGOs. We benchmark several major LLMs on this task; finding GPT-4-Turbo performs significantly better than others. We also build automatic evaluation metrics, including an efficient and accurate BERT model that outperforms finetuned LLMs, investigate how to successfully tailor vaccine messaging to CGOs, and provide actionable recommendations from this investigation.Tailor-CGO dataset and code available at: https://github.com/rickardstureborg/tailor-cgo