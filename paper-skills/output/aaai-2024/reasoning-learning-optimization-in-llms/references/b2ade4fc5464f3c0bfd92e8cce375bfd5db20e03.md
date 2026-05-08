---
title: "Evolutionary Large Language Model for Automated Feature Transformation"
source: "https://www.semanticscholar.org/paper/b2ade4fc5464f3c0bfd92e8cce375bfd5db20e03"
categories: ['machine-learning-optimization-generalization', 'reasoning-learning-optimization-in-llms']
tags: ['feature-engineering', 'automl', 'large-language-models', 'evolutionary']
venue: "AAAI 2024"
tldr: "Proposes an evolutionary large language model to automate feature transformation for downstream model performance."
---

# Evolutionary Large Language Model for Automated Feature Transformation

**Source**: [https://www.semanticscholar.org/paper/b2ade4fc5464f3c0bfd92e8cce375bfd5db20e03](https://www.semanticscholar.org/paper/b2ade4fc5464f3c0bfd92e8cce375bfd5db20e03)

**TLDR**: Proposes an evolutionary large language model to automate feature transformation for downstream model performance.

## Abstract

Feature transformation aims to reconstruct the feature space of raw features to enhance the performance of downstream models. However, the exponential growth in the combinations of features and operations poses a challenge, making it difficult for existing methods to efficiently explore a wide space. Additionally, their optimization is solely driven by the accuracy of downstream models in specific domains, neglecting the acquisition of general feature knowledge. To fill this research gap, we propose an evolutionary LLM framework for automated feature transformation. This framework consists of two parts: 1) constructing a multi-population database through an RL data collector while utilizing evolutionary algorithm strategies for database maintenance, and 2) utilizing the ability of Large Language Model (LLM) in sequence understanding, we employ few-shot prompts to guide LLM in generating superior samples based on feature transformation sequence distinction. Leveraging the multi-population database initially provides a wide search scope to discover excellent populations. Through culling and evolution, high-quality populations are given greater opportunities, thereby furthering the pursuit of optimal individuals. By integrating LLMs with evolutionary algorithms, we achieve efficient exploration within a vast space, while harnessing feature knowledge to propel optimization, thus realizing a more adaptable search paradigm. Finally, we empirically demonstrate the effectiveness and generality of our proposed method.