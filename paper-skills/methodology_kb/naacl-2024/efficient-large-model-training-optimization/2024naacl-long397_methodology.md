# Extremely Weakly-supervised Text Classification with Wordsets Mining and Sync-Denoising

**Source**: https://aclanthology.org/2024.naacl-long.397/

## [POSITIVE] Wordset-based pseudo-label generation
Representing a class with a set of words (wordsets) instead of individual seed words, and generating pseudo-labels by matching these wordsets in text.

**Delta**: outperforms SOTA by 8 points average
**Condition**: Used for pseudo-label generation in extremely weakly-supervised text classification; addresses context-dependent ambiguities of seed methods.

**Evidence**: "SetSync outperforms all existing prompt and seed methods, exceeding SOTA by an impressive average of 8 points."

## [POSITIVE] Wordsets Information Bottleneck (WIB)
A discrete-space adaptation of information bottleneck theory to mine class-relevant wordsets, combining high-frequency filtering and frequent itemset mining (FP-Growth) for optimization.

**Delta**: outperforms baseline; classifier performance improves with iterations
**Condition**: Applied in each iteration to mine new wordsets for each class; uses trained classifier predictions to identify class-specific texts.

**Evidence**: "Classifier performance gradually improved with iterations in all strategies, which demonstrating the effectiveness of our wordsets mining algorithm."

## [POSITIVE] Sync-denoising training strategy
A unified training framework that jointly optimizes noisy pseudo-labeled data and unlabeled data using weighted cross-entropy loss, with dynamic Gaussian weights for denoising.

**Delta**: outperforms supervised, self-training, FixMatch, SoftMatch, Co-Teaching, and SaFER
**Condition**: Used for classifier training in each iteration; addresses hybrid setting of semi-supervised and noisy-label learning.

**Evidence**: "Our sync-denoising training outperformed all other strategies."

## [POSITIVE] Dynamic Gaussian weight estimation
Assuming loss values of pseudo-labeled samples and prediction confidence of unlabeled samples follow two independent Gaussian distributions, with parameters updated via EMA, to compute sample weights for denoising.

**Delta**: achieves best results compared to other noise estimation methods
**Condition**: Applied during sync-denoising training; requires separate estimation for labeled and unlabeled data.

**Evidence**: "Using loss values to evaluate noise on pseudo-labeled data and using prediction confidence to evaluate unlabeled data achieved the best results."

## [POSITIVE] Majority voting for multi-class wordset matching
When a text contains wordsets from multiple classes, the class with the most occurrences (majority voting) is selected as the pseudo-label.

**Delta**: achieves relatively stable and good results in most cases
**Condition**: Used during pseudo-label generation when a text contains wordsets from multiple categories.

**Evidence**: "Majority voting achieved relatively stable and good results in most cases, followed by score accumulation."

## [POSITIVE] Iterative framework with wordset update
An iterative paradigm where pseudo-labels are generated, classifier is trained with sync-denoising, wordsets are mined via WIB, and updated for the next iteration.

**Delta**: 2nd round typically shows most significant improvement
**Condition**: Applied over multiple iterations (up to 5 in experiments); requires threshold for stopping.

**Evidence**: "We observed that the 2nd round typically showed the most significant improvement. This is mainly due to the use of mined wordsets to generate high-quality pseudo-labels, greatly improving the performance."

## [POSITIVE] High-frequency screening for wordset mining
Filtering low-frequency words per class by keeping only top Z1 words with highest term frequency in texts predicted as that class, to reduce search space for frequent itemset mining.

**Delta**: greatly reduces number of words per text, speeding up mining
**Condition**: Applied before frequent itemset mining in WIB; uses classifier predictions to aggregate class-specific texts.

**Evidence**: "Through filtering operations, we can greatly reduce the number of words in each text, thus speeding up the mining process."

## [POSITIVE] FP-Growth for frequent itemset mining
Using the FP-Growth algorithm to mine frequent itemsets (wordsets) from simplified texts of each class, outputting top Z2 itemsets with highest support.

**Delta**: better time complexity than enumeration
**Condition**: Applied after high-frequency screening in WIB; treats each simplified text as a transaction.

**Evidence**: "We choose FP-Growth for its better time complexity."

## [POSITIVE] Information bottleneck ranking of mined wordsets
Scoring candidate wordsets from FP-Growth using the WIB objective (mutual information between class and wordset, with compression penalty) and keeping top T wordsets per class.

**Delta**: system is robust to hyperparameters; high performance achieved
**Condition**: Applied after frequent itemset mining; uses representations from the last layer of the trained classifier.

**Evidence**: "Our system is robust to these hyperparameters, and these parameter selections can achieve high performance."

## [POSITIVE] Using loss value for pseudo-labeled data noise estimation
Evaluating noise level of pseudo-labeled samples based on their loss value deviation from the mean of a dynamic Gaussian distribution, down-weighting high-loss samples.

**Delta**: achieves best results compared to using prediction confidence for pseudo-labeled data
**Condition**: Applied during sync-denoising training for pseudo-labeled samples; requires EMA estimation of Gaussian parameters.

**Evidence**: "Using loss values to evaluate noise on pseudo-labeled data and using prediction confidence to evaluate unlabeled data achieved the best results."

## [POSITIVE] Using prediction confidence for unlabeled data noise estimation
Evaluating noise level of unlabeled samples based on their prediction confidence deviation from the mean of a dynamic Gaussian distribution, down-weighting low-confidence samples.

**Delta**: achieves best results compared to using loss value for unlabeled data
**Condition**: Applied during sync-denoising training for unlabeled samples; requires EMA estimation of Gaussian parameters.

**Evidence**: "Using loss values to evaluate noise on pseudo-labeled data and using prediction confidence to evaluate unlabeled data achieved the best results."
