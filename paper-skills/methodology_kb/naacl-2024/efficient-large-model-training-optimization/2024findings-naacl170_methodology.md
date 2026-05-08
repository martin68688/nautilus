# UEGP: Unified Expert-Guided Pre-training for Knowledge Rekindle

**Source**: https://aclanthology.org/2024.findings-naacl.170/

## [POSITIVE] Knowledge Rekindle Paradigm
A cyclical training paradigm that re-incorporates fine-tuned expert models into the training cycle to break through performance upper bounds without additional annotated data.

**Delta**: outperforms baseline on all 8 GLUE tasks
**Condition**: Applicable to any PLM and downstream task; particularly effective when student model capacity is larger and when labeled data is scarce.

**Evidence**: "The experimental results show that knowledge rekindle paradigm significantly outperforms traditional pre-training and fine-tuning baselines on almost all 8 NLU tasks."

## [POSITIVE] Unified Expert-Guided Pre-training (UEGP)
A framework that uses fine-tuned expert models as teacher models to guide general PLMs (student models) via knowledge distillation, combined with masked language modeling (MLM) in a multi-task learning setup.

**Delta**: UEGP-BERT-large-FT improves over BERT-large-FT by +0.87 on STS-B, +1.62 on MRPC
**Condition**: Used during the expert-guided pre-training stage; requires a pre-trained PLM as student and a fine-tuned expert as teacher.

**Evidence**: "UEGP-BERT-large-FT has improved by 1.24 compared to its teacher model BERT-base-FT, and is also superior to BERT-large-FT with the same capacity by 0.87."

## [POSITIVE] Multi-task Learning with KD and MLM
Combining knowledge distillation (KD) loss and masked language modeling (MLM) loss during expert-guided pre-training to prevent overfitting to expert knowledge and retain general language modeling capabilities.

**Delta**: UEGP-BERT-large-FT(KD+MLM) achieves 90.58 on STS-B vs 90.30 for KD-only and 89.65 for MLM-only
**Condition**: Applied during the expert-guided pre-training phase; MLM acts as a regularizer to prevent premature convergence.

**Evidence**: "The MLM objective, as a regularization term, can effectively prevent the expert-guided PLMs from overfitting expert models, slow down the occurrence of local minima, and ensure that task-specific fine-tuning can further improve performance."

## [POSITIVE] Mixture-of-Expert Guided Pre-training (MoEGP)
An extension of UEGP that uses multiple task-specific expert models simultaneously for knowledge distillation, enabling the student model to learn from multiple tasks at once.

**Delta**: MoEGP-BERT-large-FT achieves 90.80 on STS-B vs 90.58 for UEGP; 63.73 on CoLA vs 62.02; 93.92 on SST-2 vs 93.46
**Condition**: Applicable when multiple downstream tasks are available; reduces training cost compared to separate UEGP for each task.

**Evidence**: "MoE guided pre-training distills the knowledge of multiple task-specific teacher models, and the knowledge for multiple tasks can complement each other, which helps to improve the performance of PLMs on specific tasks."

## [POSITIVE] Larger Student Model Capacity
Using a larger PLM (e.g., BERT-large with 24 layers) as the student model instead of a smaller one (e.g., BERT-base with 12 layers) during expert-guided pre-training.

**Delta**: UEGP-BERT-large-FT improves over UEGP-BERT-base-FT by +0.67 on STS-B, +2.77 on CoLA, +1.15 on SST-2
**Condition**: When larger PLMs are available; larger student models have higher performance upper bounds.

**Evidence**: "The improvements are more significant when the student model capacity increases. For example, on the STS-B dataset, UEGP-BERT-base-FT has improved by 0.57 compared to BERT-base-FT, UEGP-BERT-large-FT has improved by 1.24 compared to its teacher model BERT-base-FT."

## [POSITIVE] Larger Teacher Model Size
Using a larger fine-tuned expert model (e.g., BERT-large-FT) as the teacher instead of a smaller one (e.g., BERT-base-FT).

**Delta**: UEGP-BERT-large-FT(24-24) achieves 94.15 on SST-2 vs 93.92 for (12-24); 74.36 on RTE vs 73.28
**Condition**: When larger teacher models are available; larger teachers contain richer knowledge.

**Evidence**: "The general trend is that the larger the size of teacher models, the greater the size of student models, and the more significant the performance of the knowledge rekindle paradigm."

## [POSITIVE] Task-agnostic Rekindle Data Collection
Using unlabeled corpus from public sources (e.g., Wikipedia) rather than domain-specific or task-specific data for the expert-guided pre-training stage.

**Delta**: UEGP-BERT-large-FT(KD+MLM) outperforms UEGP-BERT-large-FT(MLM) which uses only MLM on rekindle data (90.58 vs 89.65 on STS-B)
**Condition**: Applicable when task-agnostic data is readily available; reduces the need for specialized data collection.

**Evidence**: "The improvement comes from the student gaining the prior knowledge of teacher model by learning to imitate the behavior of the expert model, rather than by learning domain or task-related knowledge from extra data."

## [POSITIVE] Knowledge Distillation (KD) Objective
Using the output logits of the teacher model (fine-tuned expert) as soft targets to train the student model via MSE, KL-divergence, or hinge loss depending on the task type.

**Delta**: UEGP-BERT-large-FT(KD+MLM) achieves 90.58 on STS-B vs 89.65 for MLM-only
**Condition**: Used during expert-guided pre-training; loss type depends on task (MSE for regression, KL-divergence for classification, etc.).

**Evidence**: "The knowledge distillation objective enables PLMs to obtain prior knowledge for downstream tasks and promotes the performance improvement after further fine-tuning."

## [POSITIVE] Masked Language Modeling (MLM) as Regularization
Retaining the MLM objective during expert-guided pre-training to prevent overfitting to the teacher model's knowledge and avoid convergence to local minima.

**Delta**: UEGP-BERT-large-FT(KD+MLM) achieves 92.54 on QNLI vs 91.90 for KD-only
**Condition**: Applied jointly with KD loss during expert-guided pre-training; critical for maintaining generalization.

**Evidence**: "The MLM objective, as a regularization term, can effectively prevent the expert-guided PLMs from overfitting expert models, slow down the occurrence of local minima, and ensure that task-specific fine-tuning can further improve performance."

## [POSITIVE] Data Scarcity Scenario Application
Applying the knowledge rekindle paradigm to tasks with limited labeled data (e.g., STS-B, CoLA, RTE) where fine-tuning alone tends to converge to local minima.

**Delta**: UEGP-BERT-large-FT improves over BERT-large-FT by +0.87 on STS-B, +2.23 on CoLA, +2.17 on RTE
**Condition**: Particularly effective when labeled data is scarce; helps overcome suboptimal convergence.

**Evidence**: "The performance improvement on the STS-B, CoLA, and RTE datasets is the most significant, which suggests that the knowledge rekindle paradigm is beneficial to improve capabilities of PLMs on data scarcity scenarios."

## [POSITIVE] Pointwise and Pairwise MSE Loss for Ranking
Using pointwise and pairwise MSE loss as knowledge distillation objectives for ranking tasks to better align with fine-tuning objectives.

**Delta**: UEGP-ERNIE-48layer-FT achieves 3.584 on RE-RANK eval vs 3.526 for ERNIE-48layer-FT
**Condition**: Applied specifically to ranking tasks (e.g., search re-ranking) during expert-guided pre-training.

**Evidence**: "In the expert-guided pre-training stage, in order to better align with fine-tuning objectives, we adopt pointwise and pairwise MSE loss as the knowledge distillation loss."
