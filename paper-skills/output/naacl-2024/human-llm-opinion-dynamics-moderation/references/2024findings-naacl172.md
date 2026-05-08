---
title: "HateModerate: Testing Hate Speech Detectors against Content Moderation Policies"
source: "https://aclanthology.org/2024.findings-naacl.172/"
categories: ['llm-backdoor-attacks-defense', 'human-llm-opinion-dynamics-moderation']
tags: ['hate-speech', 'content-moderation', 'policy-testing']
venue: "NAACL 2024"
tldr: "Tests automated hate speech detectors against real-world social media content moderation policies."
---

# HateModerate: Testing Hate Speech Detectors against Content Moderation Policies

**Source**: [https://aclanthology.org/2024.findings-naacl.172/](https://aclanthology.org/2024.findings-naacl.172/)

**TLDR**: Tests automated hate speech detectors against real-world social media content moderation policies.

## Abstract

AbstractTo protect users from massive hateful content, existing works studied automated hate speech detection. Despite the existing efforts, one question remains: Do automated hate speech detectors conform to social media content policies? A platform’s content policies are a checklist of content moderated by the social media platform. Because content moderation rules are often uniquely defined, existing hate speech datasets cannot directly answer this question. This work seeks to answer this question by creating HateModerate, a dataset for testing the behaviors of automated content moderators against content policies. First, we engage 28 annotators and GPT in a six-step annotation process, resulting in a list of hateful and non-hateful test suites matching each of Facebook’s 41 hate speech policies. Second, we test the performance of state-of-the-art hate speech detectors against HateModerate, revealing substantial failures these models have in their conformity to the policies. Third, using HateModerate, we augment the training data of a top-downloaded hate detector on HuggingFace. We observe significant improvement in the models’ conformity to content policies while having comparable scores on the original test data. Our dataset and code can be found on https://github.com/stevens-textmining/HateModerate.