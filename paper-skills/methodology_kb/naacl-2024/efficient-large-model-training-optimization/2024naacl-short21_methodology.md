# On Retrieval Augmentation and the Limitations of Language Model Training

**Source**: https://aclanthology.org/2024.naacl-short.21/

## [POSITIVE] kNN Retrieval Augmentation (kNN-LM)
Augmenting a language model with k-nearest neighbors retrieval on its training data, where a datastore maps intermediate representations of prefixes to next tokens, and the final distribution is interpolated with the LM's output.

**Delta**: Reduces perplexity from 20.13 to 16.92 on WikiText; improves log-likelihood on Macondo dataset (e.g., from ~-3.5 to ~-3.17 average at 10k steps).
**Condition**: Applicable when training data contains over-specified information; effective for both GPT-2 and Mistral 7B models.

**Evidence**: "kNN augmentation consistently improves performance in this setting. ... the kNN-augmented model performs better than the vanilla model."

## [NEUTRAL] Softmax Bottleneck Investigation (Projection Experiment)
Projecting the kNN-LM distribution back into the output space of the LM's last layer (f or f∘g) via gradient descent to test if the softmax bottleneck prevents the LM from approximating the kNN distribution.

**Delta**: Projected perplexity (16.76 for f, 16.78 for f∘g) is similar to kNN-LM perplexity (16.92), with KL-divergence under 0.1.
**Condition**: Applies to analysis of GPT-2 on WikiText; shows that the last layers have sufficient capacity to generate kNN-like distributions.

**Evidence**: "The approximation of p_knnlm by the last layer f has an average perplexity similar to the perplexity of kNN-LM. ... the softmax bottleneck is not the cause of the gap."

## [NEGATIVE] Over-specification Generalization Challenge (Macondo Dataset)
A synthetic dataset where training sentences include causally irrelevant descriptors (e.g., 'who was born in [year]') that are absent at test time, testing whether LMs can generalize to non-over-specified contexts.

**Delta**: GPT-2 XL log-likelihood on test set is much lower than theoretical perfect log-likelihood (e.g., ~-3.5 vs -0.69 for 2 children); GPT-3.5 Turbo also fails to generalize.
**Condition**: Applies when training data contains redundant, causally irrelevant information; scaling model size does not solve this limitation.

**Evidence**: "The fine-tuned GPT-2 model has a likelihood much lower than the theoretical perfect likelihood. ... GPT-3.5-turbo can not generalize to a test set without over-specification."

## [POSITIVE] MLP Replacement for kNN Retrieval
Training a multi-layer perceptron (MLP) model to map datastore keys to values as a drop-in replacement for traditional kNN retrieval, reducing storage costs.

**Delta**: Reduces storage costs by over 25x; on WikiText, perplexity improves from 20.13 (LM) to 18.68 (MLP-LM) vs 16.92 (kNN-LM); on Macondo, from 19.66 to 10.76 (MLP-LM) vs 17.69 (kNN-LM).
**Condition**: Applicable as a more storage-efficient alternative to kNN augmentation; effective on both synthetic (Macondo) and real (WikiText) datasets.

**Evidence**: "Interpolating the original LM with this MLP model effectively reduces the perplexity while requiring less than 4% of storage."

## [NEGATIVE] MLP Hurdle (Last MLP Layer Slows Convergence)
Observation that the last MLP layer in the LM (f∘g) slows down convergence during early training, potentially making optimization of the encoder (enc) more difficult.

**Delta**: Models without the last MLP layer converge faster: on Macondo, log-likelihood grows faster in first 4000 steps; on WikiText, loss decreases faster at early stage.
**Condition**: Observed during early training phases (first few thousand steps) for both GPT-2 small on Macondo and WikiText.

**Evidence**: "The model's log-likelihood with the last MLP layer removed grows faster than the original model during the first 4000 steps. ... the last MLP layer slows down the convergence rate at the early phase."
