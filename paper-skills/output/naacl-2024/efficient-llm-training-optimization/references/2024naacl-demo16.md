---
title: "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions"
source: "https://aclanthology.org/2024.naacl-demo.16/"
categories: ['efficient-llm-training-optimization', 'efficient-transformer-acceleration-techniques']
tags: ['library', 'interventions', 'interpretability', 'model-editing']
venue: "NAACL 2024"
tldr: "pyvene is an open-source Python library for performing customizable interventions on PyTorch model internal states to support research in model editing, steering, robustness, and interpretability."
---

# pyvene: A Library for Understanding and Improving PyTorch Models via Interventions

**Source**: [https://aclanthology.org/2024.naacl-demo.16/](https://aclanthology.org/2024.naacl-demo.16/)

**TLDR**: pyvene is an open-source Python library for performing customizable interventions on PyTorch model internal states to support research in model editing, steering, robustness, and interpretability.

## Abstract

AbstractInterventions on model-internal states are fundamental operations in many areas of AI, including model editing, steering, robustness, and interpretability. To facilitate such research, we introduce pyvene, an open-source Python library that supports customizable interventions on a range of different PyTorch modules. pyvene supports complex intervention schemes with an intuitive configuration format, and its interventions can be static or include trainable parameters. We show how pyvene provides a unified and extensible framework for performing interventions on neural models and sharing the intervened upon models with others. We illustrate the power of the library via interpretability analyses using causal abstraction and knowledge localization. We publish our library through Python Package Index (PyPI) and provide code, documentation, and tutorials at ‘https://github.com/stanfordnlp/pyvene‘.