---
title: "Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.193/"
categories: ['llm-education-assessment-augmentation', 'topic-modeling-and-keyphrase-generation']
tags: ['distractor-generation', 'math-education', 'multiple-choice-questions']
venue: "NAACL 2024"
tldr: "LLMs are explored for automatically generating plausible incorrect answer options (distractors) for math multiple-choice questions."
---

# Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.193/](https://aclanthology.org/2024.findings-naacl.193/)

**TLDR**: LLMs are explored for automatically generating plausible incorrect answer options (distractors) for math multiple-choice questions.

## Abstract

AbstractMultiple-choice questions (MCQs) are ubiquitous in almost all levels of education since they are easy to administer, grade, and are a reliable format in assessments and practices. One of the most important aspects of MCQs is the distractors, i.e., incorrect options that are designed to target common errors or misconceptions among real students. To date, the task of crafting high-quality distractors largely remains a labor and time-intensive process for teachers and learning content designers, which has limited scalability. In this work, we study the task of automated distractor generation in the domain of math MCQs and explore a wide variety of large language model (LLM)-based approaches, from in-context learning to fine-tuning. We conduct extensive experiments using a real-world math MCQ dataset and find that although LLMs can generate some mathematically valid distractors, they are less adept at anticipating common errors or misconceptions among real students.