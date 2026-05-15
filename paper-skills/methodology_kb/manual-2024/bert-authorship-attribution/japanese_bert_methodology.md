# BERT Fine-tuning for Japanese Authorship Attribution

**Source**: local:japanese_bert.pdf

## [POSITIVE] Fine-tuning pre-trained BERT (text-only)
Fine-tuning the Japanese pre-trained BERT model (cl-tohoku/bert-large-japanese-v2) using only raw text data for authorship attribution.

**Delta**: 84% accuracy for 5 authors; 82% accuracy for 80 authors; 97% accuracy for binary Japanese speaker classification
**Condition**: When using only text data for Japanese native speakers; works better with fewer authors and for binary classification tasks.

**Evidence**: "Key findings reveal an 84% accuracy rate for identifying five authors using only text data, and a notable 82% accuracy with an expanded set of 80 authors"

## [NEGATIVE] Adding stylometric features to BERT
Incorporating four Japanese-specific stylometric features (usage rate of content words, POS bigrams, particle bigrams, comma placement) into the BERT fine-tuning pipeline by concatenating encoded features with BERT output.

**Delta**: Accuracy dropped to 53% for 25 authors (vs. higher text-only results); drastically low accuracies across all author counts
**Condition**: When adding stylometric features to BERT for Japanese AA; consistently failed across all author count patterns.

**Evidence**: "the addition of stylistic features for a set of 25 authors resulted in reduced accuracy (53%)"

## [POSITIVE] Binary classification (Japanese vs. non-Japanese)
Fine-tuning BERT to classify whether a text was written by a Japanese native speaker or a Japanese learner.

**Delta**: 97% accuracy, 80.6% F1 score
**Condition**: Binary classification task; benefits from being a simpler two-class problem.

**Evidence**: "classifying whether an individual is a Japanese native speaker... resulted in an accuracy of 97% and an F1 Score of 80%"

## [NEUTRAL] Nationality prediction (multi-class)
Fine-tuning BERT to predict the specific nationality of the author from multiple countries.

**Delta**: 61.8% accuracy, 52.0% F1 score
**Condition**: Multi-class nationality prediction; moderate performance, lower than binary native-speaker classification.

**Evidence**: "aimed to classify the nationality of authors, resulting in an accuracy of 61% and an F1 Score of 52%"

## [POSITIVE] Using Japanese pre-trained BERT (cl-tohoku/bert-large-japanese-v2)
Utilizing a domain-specific Japanese pre-trained BERT model from the Transformer library instead of a general multilingual model.

**Delta**: Enables effective fine-tuning for Japanese text; outperforms not using a Japanese-specific model (implied)
**Condition**: When fine-tuning for Japanese language tasks; provides better contextual understanding for Japanese text.

**Evidence**: "We utilized the Japanese pre-trained BERT model available from the Transformer library, named 'cltohoku/bert-large-japanese-v2'"

## [NEGATIVE] Training with limited and imbalanced data per author
Using a dataset where the number of texts per author varies widely (mean ~16, max 68, min 5) and total text length is short (average 20 tokens per line).

**Delta**: Accuracy failed to exceed 90%; lower accuracy for Japanese learners with high variability
**Condition**: When dataset has high imbalance in texts per author and short text lengths; particularly problematic for non-native speaker texts.

**Evidence**: "The bias in the number of texts per author and the limited amount of text data per author in the Japanese corpus used in this study are also considered factors that prevented accuracy from exceeding 90%"

## [NEGATIVE] Training with noisy learner data
Including compositions from Japanese learners which contain grammatical and lexical errors.

**Delta**: Lower accuracy compared to native speaker experiments (e.g., 0.0% for 5 non-Japanese authors)
**Condition**: When training on non-native speaker text with errors; introduces noise that hinders accurate classification.

**Evidence**: "One reason for the lower fine-tuning accuracy for texts by Japanese learners could be the numerous grammatical and lexical errors present in their compositions, which introduced noise"

## [NEGATIVE] Feature concatenation architecture (BERT + custom features)
Architecture where text is fed into BERT and stylometric features are encoded separately, then concatenated before the classification layer.

**Delta**: Drastically low accuracies across all patterns; failure in all experiments
**Condition**: When combining BERT with handcrafted stylometric features via concatenation; suggests mismatch between deep learning and traditional stylometric features.

**Evidence**: "Directly fine-tuning BERT with stylometric features might have been one reason for the low accuracy... extremely low accuracy was consistently observed across all patterns"

## [NEUTRAL] Using GiNZA for feature extraction
Pre-extracting Japanese stylometric features using the GiNZA NLP library before training.

**Delta**: No quantitative improvement; part of the failed stylometric feature pipeline
**Condition**: When preprocessing Japanese text for stylometric feature extraction; tool choice did not prevent failure of the overall approach.

**Evidence**: "All stylistic features were pre-extracted using GiNZA (Matsuda, 2020) and incorporated into the dataset"
