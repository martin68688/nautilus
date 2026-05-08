---
title: "Extending Input Contexts of Language Models through Training on Segmented Sequences"
source: "https://aclanthology.org/2024.findings-naacl.191/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques', 'large-language-model-evaluation-augmentation']
tags: ['long-context', 'training', 'segmentation']
venue: "NAACL 2024"
tldr: "Explores methods for adapting language models to longer input contexts by training on segmented sequences."
---

# Extending Input Contexts of Language Models through Training on Segmented Sequences

**Source**: [https://aclanthology.org/2024.findings-naacl.191/](https://aclanthology.org/2024.findings-naacl.191/)

**TLDR**: Explores methods for adapting language models to longer input contexts by training on segmented sequences.

## Abstract

AbstractEffectively training language models on longinputs poses many technical challenges. As acost consideration, languages models are pre-trained on a fixed sequence length before beingadapted to longer sequences. We explore var-ious methods for adapting models to longerinputs by training on segmented sequences andan interpolation-based method for extendingabsolute positional embeddings. We developa training procedure to extend the input con-text size of pretrained models with no architec-tural changes and no additional memory coststhan training on the original input lengths. Bysub-sampling segments from long inputs whilemaintaining their original position the model isable to learn new positional interactions. Ourmethod benefits both models trained with abso-lute positional embeddings, by extending theirinput contexts, as well as popular relative posi-tional embedding methods showing a reducedperplexity on sequences longer than they weretrained on. We demonstrate our method canextend input contexts by a factor of 4× whileimproving perplexity.