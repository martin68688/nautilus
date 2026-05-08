---
title: "LeanReasoner: Boosting Complex Logical Reasoning with Lean"
source: "https://aclanthology.org/2024.naacl-long.416/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction', 'llm-alignment-safety-detoxification']
tags: ['logical-reasoning', 'theorem-proving', 'lean']
venue: "NAACL 2024"
tldr: "Uses the Lean theorem prover to boost complex logical reasoning in language models."
---

# LeanReasoner: Boosting Complex Logical Reasoning with Lean

**Source**: [https://aclanthology.org/2024.naacl-long.416/](https://aclanthology.org/2024.naacl-long.416/)

**TLDR**: Uses the Lean theorem prover to boost complex logical reasoning in language models.

## Abstract

AbstractLarge language models (LLMs) often struggle with complex logical reasoning due to logical inconsistencies and the inherent difficulty ofsuch reasoning. We use Lean, a theorem proving framework, to address these challenges. By formalizing logical reasoning problems intotheorems within Lean, we can solve them by proving or disproving the corresponding theorems. This method reduces the risk of logical inconsistencies with the help of Lean’s symbolic solver. It also enhances our ability to treat complex reasoning tasks using Lean’s extensive library of theorem proofs. Our method achieves state-of-the-art performance on the FOLIO dataset and achieves performance near this level on ProofWriter. Notably, these results were accomplished by fine-tuning on fewer than 100 in-domain samples for each dataset