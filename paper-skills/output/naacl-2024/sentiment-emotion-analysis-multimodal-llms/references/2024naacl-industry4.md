---
title: "UINav: A Practical Approach to Train On-Device Automation Agents"
source: "https://aclanthology.org/2024.naacl-industry.4/"
categories: ['language-grounded-embodied-navigation', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['on-device', 'automation', 'ui-navigation', 'training']
venue: "NAACL 2024"
tldr: "Presents a practical approach to train on-device agents for UI automation."
---

# UINav: A Practical Approach to Train On-Device Automation Agents

**Source**: [https://aclanthology.org/2024.naacl-industry.4/](https://aclanthology.org/2024.naacl-industry.4/)

**TLDR**: Presents a practical approach to train on-device agents for UI automation.

## Abstract

AbstractAutomation systems that can autonomously drive application user interfaces to complete user tasks are of great benefit, especially when users are situationally or permanently impaired. Prior automation systems do not produce generalizable models while AI-based automation agents work reliably only in simple, hand-crafted applications or incur high computation costs. We propose UINav, a demonstration-based approach to train automation agents that fit mobile devices, yet achieving high success rates with modest numbers of demonstrations. To reduce the demonstration overhead, UINav uses a referee model that provides users with immediate feedback on tasks where the agent fails, and automatically augments human demonstrations to increase diversity in training data. Our evaluation shows that with only 10 demonstrations can achieve 70% accuracy, and that with enough demonstrations it can surpass 90% accuracy.