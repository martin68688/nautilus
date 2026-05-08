# How does Multi-Task Training Affect Transformer In-Context Capabilities? Investigations with Function Classes

**Source**: https://aclanthology.org/2024.naacl-short.15/

## [POSITIVE] Mixed Curriculum Learning
Training on tasks in order of increasing difficulty while mixing in previously seen tasks. The model is trained on partitions: first only linear, then linear+quadratic, then linear+quadratic+cubic, with tasks sampled uniformly from the current partition.

**Delta**: Outperforms both random and sequential curricula; achieves optimal MSE on quadratic where single-task models fail; uses only 1/9 of training data for comparable performance on cubic
**Condition**: When training on multiple function classes (linear, quadratic, cubic) in an ICL setting; evaluated on normalized MSE for in-context learning of function classes.

**Evidence**: "The mixed curriculum strategy provides the most benefit towards learning multiple tasks. ... the mixed curriculum model can improve data efficiency, learning harder tasks with fewer examples."

## [NEGATIVE] Sequential Curriculum Learning
Training on tasks in order of increasing difficulty without mixing: first linear, then quadratic, then cubic, each in separate partitions of training steps.

**Delta**: Performs substantially worse than mixed and random curricula; unable to learn any of the tasks effectively
**Condition**: When training on multiple function classes; likely suffers from catastrophic forgetting.

**Evidence**: "The sequential curriculum model is unable to learn any of the tasks. ... the sequential curriculum performs substantially worse (y-axis is limited in order for mixed and random curricula to be differentiated)."

## [NEUTRAL] Random Curriculum Learning
At each training step, randomly sampling a task from all K tasks with equal probability.

**Delta**: Performs comparatively worse than mixed curriculum; unable to learn quadratic function class
**Condition**: When training on multiple function classes; lacks stable periods of training for each task.

**Evidence**: "The random curriculum performs comparatively worse, whereas the sequential curriculum performs substantially worse. ... the random curriculum model performs well compared to the mixed curriculum model in linear and cubic evaluation, however it is unable to learn the quadratic function class."

## [NEGATIVE] Retrospective Head Masking
Identifying attention heads that attend to previous input tokens (retrospective heads) and masking them during evaluation to measure their importance for ICL.

**Delta**: Causes significant increase in normalized MSE compared to masking non-retrospective heads
**Condition**: When evaluating mixed curriculum models on linear, quadratic, and cubic function classes; demonstrates that specific heads are responsible for ICL capability.

**Evidence**: "Masking retrospective heads ... causes significant increase in normalized MSE compared to non-retrospective heads in the mixed curriculum model."

## [POSITIVE] Multi-Task Training (MTL) on Function Classes
Training a Transformer on multiple function classes (linear, quadratic, cubic) simultaneously rather than on a single function class.

**Delta**: 60% of mixed curriculum models converge on quadratic vs 0% of single-task models; achieves optimal MSE where single-task models fail
**Condition**: When training on quadratic function classes that are difficult for single-task models to learn; benefits from warm-up on easier tasks.

**Evidence**: "60% of mixed curriculum models converge, whereas 0% of the single-task models trained on quadratic function classes converge. ... curriculum learning can be used to learn difficult function classes that are otherwise unlearnable by single-task models."

## [POSITIVE] Data Efficiency via Curriculum Pretraining
Using a mixed curriculum model pretrained on easier tasks (linear and quadratic) to initialize training on a harder task (cubic), using fewer total examples.

**Delta**: Uses 1/9 of training examples compared to single-task cubic model; starts at 200 normalized MSE vs 450 for single-task model
**Condition**: When training on cubic function class; leverages transfer learning from easier tasks to improve initial performance and data efficiency.

**Evidence**: "The mixed curriculum model is pre-trained on 1/9 of the training examples seen by the single-task cubic model, yet the mixed curriculum model has better performance on the validation set. ... the cubic model starts at 450 normalized MSE, whereas the mixed model starts at 200 normalized MSE."

## [NEUTRAL] One-Hot Encoded Instruction (OHEI) Vector
Appending a one-hot encoded vector to the beginning of the input sequence to indicate which task is being performed, then applying a linear transformation to match the model dimension.

**Delta**: Causes minimal improvement over mixed curriculum without instruction prompting
**Condition**: When applied to mixed curriculum models for function class learning; may not be tractable due to difficulty of learning high-dimensional instruction.

**Evidence**: "Applications of the one hot encoded instruction (OHEI) vector to the mixed curriculum causes minimal improvement ... the one-hot encoded vectors may just be seen as noise."

## [NEGATIVE] Preset Instruction Vector (PI)
Appending a unique, fixed random vector sampled from an isotropic Gaussian distribution to the input sequence to indicate the task.

**Delta**: Worsens model performance in quadratic and cubic function class evaluation during test time
**Condition**: When applied to mixed curriculum models for function class learning; likely disrupts the input-output pair structure.

**Evidence**: "Application of the preset instruction (PI) vector to the mixed curriculum worsens model performance in the quadratic and cubic function class evaluation during test time. ... it may be seen as an extreme version of noise (it may disrupt the flow of xi, f(xi), confusing the model)."
