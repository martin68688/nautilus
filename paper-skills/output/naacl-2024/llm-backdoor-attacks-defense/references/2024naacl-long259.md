---
title: "Generating Attractive and Authentic Copywriting from Customer Reviews"
source: "https://aclanthology.org/2024.naacl-long.259/"
categories: ['llm-backdoor-attacks-defense', 'personalized-marketing-content-generation']
tags: ['text-generation', 'e-commerce']
venue: "NAACL 2024"
tldr: "A method generates attractive and authentic product copywriting from customer reviews."
---

# Generating Attractive and Authentic Copywriting from Customer Reviews

**Source**: [https://aclanthology.org/2024.naacl-long.259/](https://aclanthology.org/2024.naacl-long.259/)

**TLDR**: A method generates attractive and authentic product copywriting from customer reviews.

## Abstract

AbstractThe goal of product copywriting is to capture the interest of potential buyers by emphasizing the features of products through text descriptions. As e-commerce platforms offer a wide range of services, it’s becoming essential to dynamically adjust the styles of these auto-generated descriptions. Typical approaches to copywriting generation often rely solely on specified product attributes, which may result in dull and repetitive content. To tackle this issue, we propose to generate copywriting based on customer reviews, as they provide firsthand practical experiences with products, offering a richer source of information than just product attributes. We have developed a sequence-to-sequence framework, enhanced with reinforcement learning, to produce copywriting that is attractive, authentic, and rich in information. Our framework outperforms all existing baseline and zero-shot large language models, including LLaMA-2-chat-7B and GPT-3.5, in terms of both attractiveness and faithfulness. Furthermore, this work features the use of LLMs for aspect-based summaries collection and argument allure assessment. Experiments demonstrate the effectiveness of using LLMs for marketing domain corpus construction. The code and the dataset is publicly available at: https://github.com/YuXiangLin1234/Copywriting-Generation.