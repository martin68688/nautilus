# Set-Aligning Framework for Auto-Regressive Event Temporal Graph Generation

**Source**: https://aclanthology.org/2024.naacl-long.214/

## [POSITIVE] Set-Aligning Framework (SAF)
A model-agnostic framework incorporating data augmentations and set-property regularisations to address the misalignment of elements in target sequences for auto-regressive event temporal graph generation.

**Delta**: outperforms baseline
**Condition**: When fine-tuning LLMs for event temporal graph generation, especially with long documents and complex graphs.

**Evidence**: "Experimental results show that our framework surpasses existing baselines for event temporal graph generation."

## [POSITIVE] Set Property Regularisations (SPR)
A group of novel regularisations including set cardinality regularisation, duplication regularisation, and set matching regularisation (using Hausdorff distance) to mitigate penalisation from conventional text generation loss.

**Delta**: +1.5% edge F1 on NYT-test, +4.5% on NYT-human
**Condition**: Applied after initial fine-tuning iterations when the model has basic proficiency in generating correct DOT sequences.

**Evidence**: "SAF (w/o DA) achieves an improvement of approximately 1.5% on the NYT-test and 4.5% on the NYT-human datasets in terms of edge F1, demonstrating the effectiveness of SPR alone."

## [POSITIVE] Data Augmentation (Edge Order Permutation)
Introducing random permutations of set elements (edges) as augmented training examples to enforce order-invariance.

**Delta**: +3% edge F1 on NYT-test, +6% on NYT-human
**Condition**: When training with seq2seq models for set generation tasks where target sequences are order-invariant.

**Evidence**: "SAF (w/o SPR) improves upon Flan-T5-base by about 3% on the NYT-test and 6% on the NYT-human in terms of edge F1."

## [NEGATIVE] Prepending Set Cardinality to Target Sequence
Adding the cardinality of the ground-truth edge set to the beginning of the generation target to constrain over-generation.

**Delta**: -4% edge F1
**Condition**: When used in event temporal graph generation with Flan-T5-base.

**Evidence**: "Prepending the set cardinality of the ground-truth edge set to the generation target may also help constrain the generation model to avoid overgeneration. However, such attempts in our preliminary experiment led to an approximate 4% drop in edge F1 score, despite a significant reduction in the number of generated edges."

## [POSITIVE] Merging Reciprocal Relation Types
Transforming reciprocal relations (e.g., 'after' to 'before', 'is_included' to 'includes') by swapping head and tail events to simplify the label set.

**Delta**: +4.01% edge F1 (from 38.78 to 42.79)
**Condition**: When training models for event temporal relation extraction with imbalanced relation types.

**Evidence**: "The results show the model benefits from the simpler label set. Merge reciprocal achieves 42.79 edge F1 vs 38.78 with reciprocal relations on NYT-test."

## [NEUTRAL] Weak Supervision with CAEVO for Dataset Construction
Using an off-the-shelf event and temporal relation extraction tool (CAEVO) to automatically annotate a large-scale dataset from New York Times corpus.

**Delta**: enables large-scale training but introduces noise
**Condition**: When building large-scale training datasets for event temporal graph generation where human annotation is impractical.

**Evidence**: "Due to the presence of noisy labels used in fine-tuning, a major limitation of the proposed method is the inclusion of many imaginary events, trivial events, and negative expressions of events."

## [POSITIVE] Topic Modeling for Data Selection (LDA + TF-IDF-like metric)
Using Latent Dirichlet Allocation (LDA) on existing datasets to extract topics, then selecting documents with narrative-oriented descriptors using an 'event frequency × inverse-descriptor frequency' metric.

**Delta**: improves dataset quality for training
**Condition**: When constructing a weakly-supervised dataset for document-level event temporal graph generation.

**Evidence**: "We introduced additional steps in the data selection process to ensure that the selected documents contain high-quality event temporal graphs, which were not taken in Madaan and Yang (2021)."

## [POSITIVE] Using Flan-T5-base as Backbone Model
Selecting Flan-T5-base as the base model due to its encoder-decoder structure and balance of performance vs. resource consumption.

**Delta**: hits the sweet spot in terms of performance vs. resource consumption
**Condition**: When performing document-level event temporal graph generation with limited computational resources.

**Evidence**: "Flan-T5-base hits the sweet spot in terms of performance vs. resource consumption, allowing us to test more variants; its encoder-decoder structure is well-suited to document-level graph generation."

## [POSITIVE] Introducing SPR After Initial Fine-tuning Iterations
Delaying the introduction of Set Property Regularisations until after the model has learned to generate correct DOT sequences, to avoid training instability.

**Delta**: enables effective SPR application
**Condition**: When using set-property regularisations with auto-regressive language models for structured output generation.

**Evidence**: "To avoid the problems mentioned above, we introduce the SPR after a certain number of fine-tuning iterations. Once the model has acquired a basic proficiency in generating correct DOT sequences, the SPR can function as intended."

## [NEUTRAL] Beam Search Decoding (beam size=5)
Using beam search with a beam size of 5 and maximum length of 2048 to sample results during inference.

**Delta**: standard decoding strategy
**Condition**: During inference for auto-regressive text generation tasks.

**Evidence**: "We use the beam search with a beam size of 5 and a maximum length of 2048 to sample results."

## [NEGATIVE] Multi-hop Prompting with ChatGPT for Zero-shot Generation
Using a two-hop prompting strategy: first ask ChatGPT to generate events, then ask it to generate an event temporal graph based on the events and document.

**Delta**: outperformed by fine-tuned models (ChatGPT: 8.09 edge F1 on MATRES vs SAF: 15.96)
**Condition**: When attempting zero-shot event temporal graph generation with general-purpose LLMs without task-specific fine-tuning.

**Evidence**: "The results show that ChatGPT is outperformed by fine-tuned models... ChatGPT tends to approach event identification more like a summarisation task... identifying only approximately half of the events present in the target graph, resulting in low recall."
