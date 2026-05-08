---
title: "An NLP-Focused Pilot Training Agent for Safe and Efficient Aviation Communication"
source: "https://aclanthology.org/2024.naacl-industry.8/"
categories: ['speech-language-processing-multilingual', 'llm-education-assessment-augmentation']
tags: ['aviation', 'communication', 'training-agent', 'safety']
venue: "NAACL 2024"
tldr: "An NLP-based pilot training agent to improve safety and efficiency in aviation communication with air traffic controllers."
---

# An NLP-Focused Pilot Training Agent for Safe and Efficient Aviation Communication

**Source**: [https://aclanthology.org/2024.naacl-industry.8/](https://aclanthology.org/2024.naacl-industry.8/)

**TLDR**: An NLP-based pilot training agent to improve safety and efficiency in aviation communication with air traffic controllers.

## Abstract

AbstractAviation communication significantly influences the success of flight operations, ensuring safety of lives and efficient air transportation. In day-to-day flight operations, air traffic controllers (ATCos) would timely communicate instructions to pilots using specific phraseology for aircraft manipulation . However, pilots, originating from diverse backgrounds and understanding of English language, have struggled with conforming to strict phraseology for readback and communication in the live operation, this problem had not been effectively addressed over the past decades. Traditionally, aviation communication training involved expensive setups and resources, often relying on human-in-the-loop (HIL) air traffic simulations that demand allocating a specific environment, domain experts for participation, and substantial amount of annotated data for simulation. Therefore, we would like to propose an NLP-oriented training agent and address these challenges. Our approach involves leveraging only natural language capabilities and fine-tuning on communication data to generate instructions based on input scenarios (keywords). Given the absence of prior references for this business problem, we investigated the feasibility of our proposed solution by 1) generating all instructions at once and 2) generating one instruction while incorporating conversational history in each input. Our findings affirm the feasibility of this approach, highlighting the effectiveness of fine-tuning pre-trained models and large language models in advancing aviation communication training.