---
title: Curriculum Learning Improves Multi-Task and Instruction Tuning Performance
confidence: HIGH
papers: [2024findings-naacl.82, 2024naacl-short.15, 2024naacl-srw.7]
---

# Curriculum Learning Improves Multi-Task and Instruction Tuning Performance

Multiple papers demonstrate that organizing training data in a curriculum from simple to complex, with interleaving of different subjects/tasks, consistently improves performance over random ordering or blocking strategies.

**CORGI** (2024findings-naacl.82) organizes instruction data in a sequence progressing from simple to complex based on Bloom's Taxonomy (Remember → Understand → Apply), interleaving different subjects while globally increasing cognitive difficulty. Interleaving outperforms blocking: ΔMMLU +2.75, ΔARC +2.39, ΔPIQA +1.14, ΔOpenbookQA +3.8 compared to blocking. Overall gains of +4.76 on TruthfulQA, +2.98 on MMLU, +2.8 on OpenbookQA compared to random shuffling. The curriculum also mitigates negative impacts of noisy synthetic data, turning performance decreases into increases (ΔMMLU -0.31 → +2.75).

**Mixed Curriculum Learning** (2024naacl-short.15) trains on function classes (linear, quadratic, cubic) in order of increasing difficulty while mixing in previously seen tasks. The mixed curriculum achieves optimal MSE on quadratic where single-task models fail, and uses only 1/9 of training data for comparable performance on cubic. 60% of mixed curriculum models converge on quadratic vs 0% of single-task models. Sequential curriculum (no mixing) performs substantially worse due to catastrophic forgetting.

**Progressive Difficulty Multitask Learning** (2024naacl-srw.7) trains simultaneously on subtasks of increasing difficulty (e.g., coarse to fine label granularity) and shows improvements across all tasks tested (sentiment analysis, text classification, unit segmentation, syllogistic reasoning). For syllogistic reasoning with GPT-3.5, Progressive Chain-of-Thought prompting achieves +5.17% accuracy (72.99% → 78.16%).

## Papers & Evidence
- `2024findings-naacl.82` (CORGI): "The findings of our study reveal that substantial improvements in performance can be achieved through the mere application of curriculum ordering to instruction data—achieving gains of +4.76 on TruthfulQA, +2.98 on MMLU." — Interleaved curriculum with Bloom's Taxonomy.
- `2024naacl-short.15` (Mixed Curriculum): "The mixed curriculum strategy provides the most benefit towards learning multiple tasks... the mixed curriculum model can improve data efficiency, learning harder tasks with fewer examples." — Mixed curriculum for function classes.
- `2024naacl-srw.7` (Progressive Difficulty): "We found that by training the model on both the main task and its simplified versions simultaneously, the performance of the model improved in all cases." — Progressive difficulty multitask learning.

## Actionable Guidance
Use an interleaved curriculum (not sequential blocking) that progresses from simple to complex while mixing previously seen tasks/subjects. For instruction tuning, organize data by cognitive difficulty (Bloom's Taxonomy levels) and interleave subjects. For multi-task learning, train on simplified subtasks alongside the main task. Avoid sequential curricula that cause catastrophic forgetting.

**Condition**: When training LLMs on multi-domain instruction data or multi-task learning scenarios where tasks vary in difficulty.
**Confidence**: HIGH — Three papers with different task domains and model architectures all show consistent benefits from curriculum learning.
