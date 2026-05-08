# VertAttack: Taking Advantage of Text Classifiers’ Horizontal Vision

**Source**: https://aclanthology.org/2024.naacl-long.41/

## [POSITIVE] VertAttack: Vertical Word Transformation
Rewriting selected words vertically (one character per line) instead of horizontally, exploiting classifiers' inability to read vertical text.

**Delta**: Drops RoBERTa accuracy on SST2 from 94% to 13% (81 point drop); average drop of 83 points on AG News, 80 on SST2, 74 on CoLA, 56 on QNLI, 71 on Rotten Tomatoes.
**Condition**: When the feedback classifier is the same as the target classifier (white-box setting).

**Evidence**: "VertAttack is able to greatly drop the accuracy of 4 different transformer models on 5 datasets. For example, on the SST2 dataset, VertAttack is able to drop RoBERTa’s accuracy from 94 to 13%."

## [POSITIVE] Greedy Word Selection via Removal
Selects the word to transform by removing each word one at a time and measuring the drop in classification probability; the word causing the largest drop is chosen.

**Delta**: Enables attack to achieve up to 90 point drops in accuracy (e.g., BERT on AG News from 94.2% to 4.7%).
**Condition**: Black-box access to classifier probabilities; used in the word selection step of VertAttack.

**Evidence**: "The word that causes the highest drop in probability is chosen as the word to be transformed."

## [POSITIVE] Iterative Attack with Repeated Selection
After transforming a word, the classifier is re-queried; if misclassification is not achieved, the algorithm repeats, removing already-selected words from candidate pool.

**Delta**: Allows attack to continue until misclassification, contributing to large accuracy drops (e.g., 94% to 13% on SST2).
**Condition**: When initial single-word transformation does not cause misclassification.

**Evidence**: "If not, then the algorithm repeats, however, this time the words that have been selected already are removed as candidates during the selection step."

## [NEUTRAL] Width Constraint for Practicality
Limits the number of words transformed at a time to a fixed width (e.g., 10 words), combining all text at the end.

**Delta**: No quantitative delta reported; described as a practicality measure.
**Condition**: Applied when processing long texts to manage computational complexity.

**Evidence**: "Finally, we add a width constraint to the algorithm for practicality. The transformation is only run on that width (number of words) at a time and all text is combined at the end."

## [POSITIVE] Transferability of Attack (Different Feedback Classifier)
Using a different classifier for feedback than the target classifier (black-box transfer setting).

**Delta**: Average drop of 51 points on CoLA; drops around 40 points on SST2, AG News, Rotten Tomatoes; 25 points on QNLI.
**Condition**: When the attacker does not have access to the target classifier's internal feedback.

**Evidence**: "Though not as strong, we find VertAttack to be successful even in cases of transferability. In the most effective case (the CoLA datasets), the transfer attacks cause an average drop of 51 points."

## [POSITIVE] Chaff Insertion to Defeat Reverse Defense
Randomly inserting alphabet characters (with probability p) instead of whitespace in vertical lines to hinder reverse-engineering of the attack.

**Delta**: Drops reverse defense accuracy from ~84% to as low as ~40% (e.g., BERT at p=30%).
**Condition**: When the classifier attempts to reverse-engineer VertAttack by recombining vertical characters.

**Evidence**: "This enhancement hinders the ability to reverse the attack. Reverse is not able to identify non-perturbed characters. We see that accuracy drops from 84 to ... in the case of BERT."

## [NEGATIVE] Simple Whitespace Removal Defense
Removing extraneous whitespace and limiting to one space between tokens as a preprocessing defense.

**Delta**: Raises Albert accuracy from 14.7% to 29.7% when VertAttack does not use the same preprocessing; but drops to 13.6% when VertAttack adapts.
**Condition**: When the classifier uses simple whitespace removal and VertAttack does not adapt its feedback to match.

**Evidence**: "The simple preprocessing is able to raise Albert’s classification accuracy from 14.7 to 29.7."

## [NEGATIVE] Word Segmentation Defense
Using a text segmentation library to remove whitespace and recombine words before classification.

**Delta**: Raises Albert accuracy from 14.7% to 49.2% when VertAttack does not adapt; drops to 4.7% when VertAttack adapts.
**Condition**: When the classifier uses word segmentation and VertAttack does not adapt its feedback.

**Evidence**: "The word segmentation approach raises it even higher (to 49.2)."

## [NEGATIVE] Reverse-Engineering Defense
Classifier attempts to recombine vertical characters into words by splitting on newlines and reconstructing original text lines.

**Delta**: Increases accuracy from 6-24% to 84-87% on Rotten Tomatoes.
**Condition**: When the classifier knows the VertAttack algorithm and applies the reverse transformation.

**Evidence**: "The algorithm is able to strongly combat VertAttack, increasing the accuracy from 6 - 24 to 84 - 87."

## [NEUTRAL] OCR Pipeline for Image-Based Text
Converting modified text to an image and then using OCR to extract text before classification.

**Delta**: Accuracy increases in some cases (e.g., Albert from 13.6% to 35.7%) but remains below majority class baseline (53.3%).
**Condition**: When text is presented as an image and OCR is used to extract it.

**Evidence**: "For OCR, we see accuracy increase in the cases when the target and feedback classifier are the same. ... All accuracies are below the simple majority class baseline of 53.3."

## [POSITIVE] Human Readability Verification via Crowdsourcing
Using crowdworkers to label perturbed texts to verify that meaning is preserved for humans.

**Delta**: Humans correctly label 77% of perturbed texts vs. 81% of original texts (only 4 point drop).
**Condition**: Applied to evaluate the attack's success in preserving human understanding.

**Evidence**: "Humans were able to identify sentiment correctly, 77% of the time, far greater than the 7 - 26% of the automated classifiers."
