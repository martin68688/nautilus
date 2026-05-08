---
title: "ScriptMix: Mixing Scripts for Low-resource Language Parsing"
source: "https://aclanthology.org/2024.naacl-long.357/"
categories: ['speech-language-processing-multilingual']
tags: ['low-resource', 'parsing', 'script-adaptation']
venue: "NAACL 2024"
tldr: "Introduces ScriptMix, a method to adapt multilingual PLMs to unseen scripts by mixing them with seen scripts for low-resource language parsing."
---

# ScriptMix: Mixing Scripts for Low-resource Language Parsing

**Source**: [https://aclanthology.org/2024.naacl-long.357/](https://aclanthology.org/2024.naacl-long.357/)

**TLDR**: Introduces ScriptMix, a method to adapt multilingual PLMs to unseen scripts by mixing them with seen scripts for low-resource language parsing.

## Abstract

AbstractDespite the success of multilingual pretrained language models (mPLMs) for tasks such as dependency parsing (DEP) or part-of-speech (POS) tagging, their coverage of 100s of languages is still limited, as most of the 6500+ languages remains “unseen”. To adapt mPLMs for including such unseen langs, existing work has considered transliteration and vocabulary augmentation. Meanwhile, the consideration of combining the two has been surprisingly lacking. To understand why, we identify both complementary strengths of the two, and the hurdles to realizing it. Based on this observation, we propose ScriptMix, combining two strengths, and overcoming the hurdle.Specifically, ScriptMix a) is trained with dual-script corpus to combine strengths, but b) with separate modules to avoid gradient conflict. In combining modules properly, we also point out the limitation of the conventional method AdapterFusion, and propose AdapterFusion+ to overcome it. We empirically show ScriptMix is effective– ScriptMix improves the POS accuracy by up to 14%, and improves the DEP LAS score by up to 5.6%. Our code is publicly available.