# Authorship Attribution with Transformers

**Source**: https://aclanthology.org/2023.acl-srw.44

## [POSITIVE] GAN-BERT for Authorship Attribution
Using a GAN-BERT model (combining BERT with a semi-supervised GAN) for authorship attribution, trained only on labeled data with the discriminator classifying over N+1 classes (N authors + 1 fake class).

**Delta**: Accuracy and F1 scores over 0.88
**Condition**: When applied to late 19th-century novels with limited and imbalanced data.

**Evidence**: "we employed the GANBERT model along with data sampling strategies to fine-tune a transformer-based model for authorship attribution...yielding performance over 0.88 accuracy and F1 scores."

## [POSITIVE] Transfer Learning from 20-Author Base Model
Pre-training a GAN-BERT model on a 20-author dataset, then fine-tuning on smaller author subsets (2 to 18 authors).

**Delta**: Substantially improved performance, especially for increasing number of authors
**Condition**: When training on smaller author subsets (2 to 18 authors) after pre-training on 20 authors.

**Evidence**: "Transfer learning has substantially improved the model’s performance, especially for the increasing number of authors."

## [POSITIVE] Random Sampling of Author Combinations
Instead of manually selecting author subsets, randomly sampling 50 or 100 different author combinations for each n-author case to reduce bias.

**Delta**: Achieves higher accuracy (>0.97) compared to manual sampling
**Condition**: When evaluating model robustness across different author combinations.

**Evidence**: "Compared to the manually selected author sample sets, 50 and 100 random sampling achieves a higher accuracy for all the author cases, precisely more than 0.97% of accuracy."

## [NEGATIVE] Manual Selection of Author Sample Sets
Selecting 5 specific author combinations manually for each n-author case (2 to 18 authors).

**Delta**: Lower and more variable performance compared to random sampling
**Condition**: When using small, non-random author subsets; introduces bias.

**Evidence**: "Analysing the model with manually selected author sample sets may fail to describe the results and any trends due to the bias factors...manual samples and random samples show clear distinction with increasing the number of authors in the dataset."

## [NEGATIVE] Increasing Text Sample Size per Novel
Drawing larger numbers of 512-word text chunks (samples) from each novel, ranging from 5 to 35 samples per novel.

**Delta**: Performance drops from 0.92 F1 (5 samples) to 0.82 F1 (20 samples)
**Condition**: When sample size exceeds 5 per novel for 18-author classification; may cause overfitting or high variance.

**Evidence**: "increasing the sample size has a negative impact on the model’s performance across all sample sets for the 18-author model...the larger text samples from novels only sometimes lead to better performance."

## [POSITIVE] Pre-Screening Authors with Distribution and Filtering Parameters
Qualitative analysis selecting authors based on distribution (gender, genre, ethnicity) and filtering (publication period, number of novels, language, etc.) parameters.

**Delta**: Enables creation of a balanced, validated dataset
**Condition**: When curating a domain-specific literary dataset for authorship attribution.

**Evidence**: "We performed pre-screening on the authors before collecting the dataset, which is, to the best of our knowledge, the first attempt to perform a qualitative analysis on the literary domain for authorship attribution."

## [POSITIVE] Leave-n-Out Dataset Splitting with Stratified Train-Test-Validation
Splitting the dataset into train (80%), test (10%), validation (10%) sets stratified by author, using a leave-n-out method to create different author subsets.

**Delta**: Ensures uniform distribution of novels per author and distinct test data
**Condition**: When creating multiple author subsets for controlled experiments.

**Evidence**: "We split train-test-validation (80:10:10) sets, stratified by author ids, for each sample set considered for the experiments...The stratified split in the train-test-validation ensured a uniform distribution of novels per author, and the test data are distinct from the training data."

## [POSITIVE] Using Only Labeled Data in GAN-BERT (Modified from Semi-Supervised)
Training the GAN-BERT model with only labeled data (instead of semi-supervised with unlabeled data), using the generator to create fake samples for the N+1th class.

**Delta**: Enables authorship obfuscation detection
**Condition**: When labeled data is available and detecting fake/obfuscated writing is desired.

**Evidence**: "In contrast to GAN-BERT...which utilises a semi-supervised GAN model with labelled and unlabeled data, we train the GAN-BERT model with labelled data only...The discriminator is suitable for detecting authorship obfuscation and forgery since it is trained with fake samples similar to the original author-written texts."

## [POSITIVE] Uniform Dataset with 20 Novels per Author
Filtering authors to those with at least 20 novels and using exactly 20 novels per author to ensure balanced representation.

**Delta**: Enables fair comparison across authors
**Condition**: When comparing multiple authors with varying corpus sizes.

**Evidence**: "We use a normalised dataset of 20 novels per author to avoid dataset imbalance."

## [NEGATIVE] Baseline Comparison with Stylometric Features
Using stylometric features (e.g., from Sari et al., 2018) as a baseline model for authorship attribution.

**Delta**: Accuracy as low as 0.14 on IMDB20 dataset
**Condition**: When applied to IMDB20, Blog20, or literary datasets; performs poorly compared to neural models.

**Evidence**: "Using stylometric features performed the worst with an accuracy of 0.14 on the IMDB20 dataset."

## [NEUTRAL] Baseline Comparison with Character N-gram
Using character n-gram features as a baseline model.

**Delta**: Accuracy 0.69 (IMDB20), 0.23 (Blog20), 0.94 (20-authors), 0.95 (18-authors)
**Condition**: Performs moderately on literary datasets but poorly on blog data.

**Evidence**: "Character Ngram...0.69, 0.23, 0.94, 0.95"

## [NEUTRAL] Baseline Comparison with Word-level TF-IDF
Using word-level TF-IDF features as a baseline model.

**Delta**: Accuracy 0.97 (IMDB20), 0.47 (Blog20), 0.91 (20-authors), 0.90 (18-authors)
**Condition**: Performs well on IMDB and literary datasets but poorly on blog data.

**Evidence**: "Word level TF-IDF...0.97, 0.47, 0.91, 0.90"

## [POSITIVE] Baseline Comparison with BERTAA
Using BERTAA (fine-tuned BERT for authorship attribution) as a baseline model.

**Delta**: Accuracy 0.97 (IMDB20), 0.62 (Blog20), 0.99 (20-authors), 0.99 (18-authors)
**Condition**: Outperforms proposed GAN-BERT on literary datasets; strong baseline.

**Evidence**: "BERTAA...0.97, 0.62, 0.99, 0.99"
