# Deep Learning for Authorship Attribution

**Source**: local:arxiv2510.pdf

## [POSITIVE] Predictive Comparison
A stylometric method where a GPT-2 model is trained from scratch on one author's works, and the cross-entropy loss on held-out text is used as a measure of stylistic similarity. Lower loss indicates the text is more similar to the training author's style.

**Delta**: 100% classification accuracy
**Condition**: Applied to eight classic authors writing in English, using full books as training data.

**Evidence**: "Across every author we considered, and for every random seed, models trained and tested on the same author always yield smaller losses than models trained on one author and tested on another. Indeed, we achieve perfect (100%) classification accuracy when matching authors with held-out texts by labeling the held-out text according to which model produces the smallest loss."

## [POSITIVE] Training to a fixed loss threshold
Instead of training for a fixed number of epochs, models are trained until the cross-entropy loss falls to 3.0 or lower, enabling fair comparison across authors.

**Delta**: Enables fair comparison across authors
**Condition**: Used for all model training in the main experiment and ablation studies.

**Evidence**: "Training to a fixed loss threshold (e.g., as opposed to training for a fixed number of epochs) enables us to fairly compare model performance across authors, which is the central component of our stylometric analyses."

## [POSITIVE] Standardized token budget per author
To ensure fair comparisons, the number of training tokens per author is standardized by truncating each author's corpus to a fixed budget (643,041 tokens), determined by the smallest total token count among authors after removing the longest book.

**Delta**: Ensures fair comparisons across authors
**Condition**: Applied to all eight authors in the main experiment.

**Evidence**: "To ensure fair comparisons across authors, we standardize the number of training tokens per author by truncating each author's corpus. This token budget is determined by removing the longest book from each author's set and then taking the smallest of the (remaining) total token counts. For our dataset, this yields a fixed training token budget of 643,041 tokens."

## [POSITIVE] Contiguous sub-sequence sampling with proportional lengths
From each book in the training corpus, a contiguous sub-sequence is sampled with length proportional to the book's original length, and starting position chosen uniformly at random. These are then shuffled and concatenated into a single training sequence.

**Delta**: Enables robust training across 10 random seeds
**Condition**: Used for constructing training data for each author in the main experiment.

**Evidence**: "The length of the sub-sequence sampled from book i is proportional to its original length... The starting position of each sub-sequence is chosen uniformly at random, ensuring the sample fits within the book's bounds. Finally, we shuffle and then concatenate the sampled sub-sequences from each book, resulting in a single 643,041-token training sequence for each author. This process is repeated for each of 10 random seeds, yielding 10 different training corpora for each author."

## [POSITIVE] GPT-2 architecture with custom settings
A GPT-2 model with a context window of 1024 tokens, embedding dimension of 128, 8 transformer layers, and 8 attention heads per head, trained from scratch using the AdamW optimizer with a learning rate of 5e-5.

**Delta**: Achieves perfect classification accuracy
**Condition**: Applied to all models in the study.

**Evidence**: "For each author, we train GPT-2 language models from scratch using the GPT2LMHeadModel class from the Hugging Face Transformers library with custom architecture settings: a context window of 1024 tokens, an embedding dimension of 128, 8 transformer layers, and 8 attention heads per layer."

## [POSITIVE] Ablation: content-word-only corpora
All function words are replaced with a special <FUNC> token, leaving only content words, to study the contribution of vocabulary choice and thematic content to stylometric signatures.

**Delta**: Reliably learned author-specific patterns for 6 of 8 authors, but significantly less effective than intact texts (t(11.77)=3.21, p=7.68e-3)
**Condition**: Applied to all eight authors as an ablation study.

**Evidence**: "Models trained on content-word-only corpora reliably learned author-specific patterns for 6 of the 8 authors... However, models trained only on content words were significantly less effective at distinguishing authors than models trained on the intact texts (t(11.77) = 3.21, p = 7.68 × 10−3)."

## [POSITIVE] Ablation: function-word-only corpora
All content words are replaced with a special <CONTENT> token, leaving only function words, to study the contribution of syntactic and grammatical patterns.

**Delta**: Reliably learned author-specific patterns for 5 of 8 authors, significantly less effective than intact texts (t(8.36)=4.82, p=1.15e-3)
**Condition**: Applied to all eight authors as an ablation study.

**Evidence**: "Models trained on function-word-only corpora reliably learned author-specific patterns for 5 of the 8 authors... These models were also significantly less effective at distinguishing authors than models trained on the intact texts (t(8.36) = 4.82, p = 1.15 × 10−3)."

## [NEGATIVE] Ablation: part-of-speech-only corpora
Each word is replaced with its corresponding part-of-speech tag to study the contribution of higher-level syntactic patterns while abstracting away from specific word choices.

**Delta**: Reliably learned author-specific patterns for only 3 of 8 authors; average t-values not reliably greater than zero (t(9)=1.616, p=0.141)
**Condition**: Applied to all eight authors as an ablation study.

**Evidence**: "Models trained on part-of-speech-only corpora reliably learned author-specific patterns for just 3 of the 8 authors... By the final training epoch, the average t-values across all models and held-out texts were not reliably greater than zero (t(9) = 1.616, p = 0.141). These models were significantly less effective at distinguishing authors than models trained on the intact texts (t(7.36) = 5.72, p = 6.01 × 10−4)."

## [POSITIVE] Stylometric distance metric
A symmetrized measure of stylistic distance between authors, defined as the average of the normalized cross-entropy losses: d(i,j) = 0.5 * (L_j(i) + L_i(j)), where L_j(i) is the loss of author i's text on author j's model minus the baseline loss.

**Delta**: Produces meaningful clustering of stylistically similar authors (e.g., Baum and Thompson)
**Condition**: Applied to all eight authors in the main experiment.

**Evidence**: "Let L_j(i) denote the average loss of a work of author i for a model trained on author j... Then define the LLM-based stylometric distance, d(i, j) = 1/2 [L_j(i) + L_i(j)]. Thus, Figure 4 is a visualization of the relative 'distances' among our author set."

## [NEUTRAL] Multidimensional scaling (MDS) visualization
Projection of the pairwise stylometric distances into a 3D space using multidimensional scaling applied to the correlations between rows of the loss matrix.

**Delta**: Provides a provocative demonstration
**Condition**: Applied to the main experiment results.

**Evidence**: "We also project the losses into a 3D space using multidimensional scaling (MDS; Kruskal, 1964) applied to the pairwise correlations between rows of the loss matrix... For the purposes of our present work, however, we provide the plot solely as a provocative demonstration."

## [POSITIVE] Predictive attribution of contested work
Applying the predictive comparison method to a contested work (the 15th Oz book) by comparing losses from models trained on Baum and Thompson's undisputed works.

**Delta**: Confirms Thompson's authorship, consistent with traditional stylometric analyses
**Condition**: Applied to the 15th Oz book attribution problem.

**Evidence**: "We find lower loss for the Thompson-trained model than for the Baum-trained model, indicating that the contested book is indeed more similar to Thompson's writing style than to Baum's... confirming Thompson's authorship in agreement with traditional stylometric analyses (Binongo, 2003)."

## [POSITIVE] Cross-domain robustness check
Testing models on non-Oz books by Baum and Thompson to verify that predictive comparison is robust to thematic differences within the same author's writings.

**Delta**: Lower losses for the correct author in each case
**Condition**: Applied to Baum and Thompson's non-Oz books.

**Evidence**: "We see lower losses for the correct author in each case, demonstrating that predictive comparison is robust to thematic differences within the same author's writings."
