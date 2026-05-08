# RedCoast: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs

**Source**: https://aclanthology.org/2024.naacl-demo.14/

## [POSITIVE] Automatic Tensor Parallelism via Two Rules
Two simple rules to automatically generate tensor parallel strategies for any transformer architecture: (1) For fully-connected layers, alternate splitting parameters along dimension 1 and dimension 0. (2) For attention layers, split Q, K, V along dimension 0 and split output projection matrix O along dimension 1.

**Delta**: outperforms FSDP, close to Alpa (state-of-the-art)
**Condition**: When applying tensor parallelism to any LLM architecture; users only need to specify number of shards.

**Evidence**: "The observed throughput reveals that Redco surpasses FSDP and is close to Alpa, the state-of-the-art model parallel tool."

## [POSITIVE] Pipeline Design via Three Functions
Users define only three functions (collate, loss, predict) to specify an ML pipeline; Redco handles all low-level execution details like data parallelism, multi-host processing, checkpointing, randomness control, and logging.

**Delta**: 100-200 lines of code vs. much larger official implementations
**Condition**: Applicable to diverse ML algorithms from basic language modeling to meta-learning and reinforcement learning.

**Evidence**: "The majority of these paradigms can be efficiently implemented using Redco with only 100 to 200 lines of code."

## [POSITIVE] Standardized Checkpointing (Dict of Tensors)
Model parameters and optimizer states are saved as dictionaries of tensors, independent of distributed training configuration, unlike Megatron's custom format.

**Delta**: Simplifies loading and utilization without Redco installation
**Condition**: When saving/loading checkpoints in distributed training.

**Evidence**: "This approach is independent of the specific distributed training configurations, offering the advantage of simplicity in loading and utilization, even without Redco installation."

## [POSITIVE] Lightweight Dependency (No Package Modification)
Redco is built on top of Jax and Flax without modifying any existing packages, supporting a wider range of jax, flax, and CUDA versions compared to tools like Alpa.

**Delta**: Supports newer jaxlib 0.4.32 and CUDA 12.2 vs Alpa's limitation to jaxlib 0.3.22 and CUDA 11.3
**Condition**: When deploying in environments with varying CUDA/Jax versions.

**Evidence**: "Redco is developed on top of Jax and Flax, without any modification to existing packages. Consequently, Redco is able to support a wider range of versions of jax, flax, and CUDA."

## [POSITIVE] Automatic Multi-Host Support
Redco automatically handles data allocation across nodes, gradient aggregation, and multi-host processing without requiring users to modify their pipeline code.

**Delta**: No additional coding effort for multi-host environments
**Condition**: When scaling training across multiple hosts or clusters.

**Evidence**: "Redco offers automatic support for multi-host environments and demonstrates compatibility with various platforms, including SLURM, XManager, as well as bare nodes interconnected via IP addresses."

## [POSITIVE] Implementation via jax.pjit
Tensor parallelism is implemented on top of jax.pjit, which compiles the computational graph and merges operations on the same device to reduce unnecessary communication overhead.

**Delta**: Enables efficient distributed execution
**Condition**: When executing tensor-parallel computation on GPUs/TPUs.

**Evidence**: "This function compiles the computational graph and it merges operations on the same device to reduce unnecessary communication overhead."

## [NEUTRAL] Customizable Sharding Strategies
In addition to automatic sharding strategy generation, Redco allows users with more advanced strategies to customize and execute their own approaches.

**Delta**: Provides flexibility without quantified improvement
**Condition**: When users have domain expertise and want to override default sharding.

**Evidence**: "Moreover, in addition to automatically generating sharding strategies, Redco also enables their customization. This allows users with more advanced strategies to execute their approaches."
