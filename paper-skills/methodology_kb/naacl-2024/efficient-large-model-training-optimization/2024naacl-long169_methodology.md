# When Life Gives You Lemons, Make Cherryade: Converting Feedback from Bad Responses into Good Labels

**Source**: https://aclanthology.org/2024.naacl-long.169/

## [POSITIVE] Satisfaction Classifier with Next Human Response
A classifier trained to predict binary satisfaction labels (good/bad) for bot replies, using the dialogue context, the bot reply, and the next human response as input.

**Delta**: Balanced accuracy improved from ~75% to ~95% on FITS; balanced F1 improved from 64.77 to 71.24 on DEMO (zero-shot).
**Condition**: When the next human response is available (e.g., free-form textual feedback or natural continuation).

**Evidence**: "We find the balanced accuracy of detecting satisfaction using only the dialogue context and the bot response itself is ∼75% on FITS. It is significantly improved to ∼95% by including the next human message in the input."

## [POSITIVE] Reply Corrector with Free-form Textual Feedback
A model trained to convert bad bot replies into good ones, conditioned on the dialogue context, the bad reply, and the human's free-form textual feedback.

**Delta**: F1 improved from 17.10 to 21.41 on validation set when using free-form textual feedback vs. without it.
**Condition**: When free-form textual feedback is available for bad replies.

**Evidence**: "As expected, adding free-form textual feedback on what went wrong improves the reply corrector’s performance."

## [POSITIVE] Augmenting Reply Corrector with Self-Correction Pairs
Training the reply corrector not only on human-written gold corrections but also on 'self-correction' pairs where a bad reply is followed by a good bot reply (either liked by humans or predicted as good).

**Delta**: F1 improved from 17.10 to 21.41 on validation set.
**Condition**: When self-correction pairs (bad reply followed by a good reply) are available in the data.

**Evidence**: "We found that augmenting with these 'self-corrections' improves the F1 from 17.10 to 21.41."

## [POSITIVE] Reranking-based Correction Selection
Generating multiple correction candidates (e.g., 60) with the reply corrector, then scoring each with the satisfaction classifier and selecting the one with the highest probability of being good.

**Delta**: 99.96% of bad responses had at least one generated correction predicted as satisfactory.
**Condition**: When multiple candidate corrections can be generated and a satisfaction classifier is available for scoring.

**Evidence**: "We first use the reply corrector to generate many correction candidates (60 in our experiments). Then we concatenate the original context with the correction candidates and feed them into the satisfaction classifier... Finally, we select the top one with the highest probability output by the classifier as the final correction."

## [POSITIVE] Selecting Correctable Cases via Cosine Similarity
Filtering bad responses to only include those where the free-form textual feedback has high cosine similarity with the next bot reply (using SentenceBERT), indicating the feedback is constructive and easy to correct.

**Delta**: F1 improved from 18.0 to 18.5 on test unseen set compared to using all predicted corrections.
**Condition**: When free-form textual feedback is available and a similarity threshold can be tuned on a validation set.

**Evidence**: "Compared to naively augmenting with all predicted corrections, we see gains on valid and unseen test F1 (18.5 vs. 18.0)."

## [POSITIVE] JUICER Data Augmentation Pipeline
A four-step framework: (1) train satisfaction classifier and reply corrector on limited labeled data; (2) predict labels for unlabeled turns; (3) correct bad replies using the reply corrector; (4) retrain the dialogue model on augmented data (good + corrected replies).

**Delta**: F1 improved from 15.3 to 18.5 on test unseen set; good response rate improved from 33.2% to 41.9% in human evaluation.
**Condition**: When sparse binary/gold feedback is available but free-form textual feedback is relatively dense.

**Evidence**: "JUICER yields significant gains over the baseline transformer BB2 3B in both automatic evaluations and human evaluations."

## [POSITIVE] DIRECTOR for Utilizing Both Positive and Negative Responses
A unified decoder-classifier model that jointly trains on language modeling (positive responses) and classification (positive vs. negative responses), using the classifier head to penalize negative tokens during generation.

**Delta**: Good response rate improved from 41.9% (JUICER without DIRECTOR) to 45.5% (JUICER+DIRECTOR); human rating improved from 3.06 to 3.34.
**Condition**: When both positive and negative examples (gold or predicted) are available for training.

**Evidence**: "DIRECTOR utilizes both the (predicted) binary feedback signal and textual feedback signal to penalize negative responses. Applying it improves the results further over standard JUICER."

## [POSITIVE] Supervised Reply Corrector vs. Prompt-based Corrector
Using a fine-tuned supervised model (R2C2) as the reply corrector instead of prompting a large language model with instructions.

**Delta**: JUICER models (supervised) achieved F1=16.7 vs. prompt-based baseline F1=14.2 on validation set.
**Condition**: When a moderate amount of labeled correction data is available to fine-tune a supervised model.

**Evidence**: "Compared to the prompt-based reply corrector baseline (Scheurer et al., 2022), all the JUICER models perform better in automatic evaluations."

## [POSITIVE] Training on Gold Corrections from 20% Data
Fine-tuning the dialogue model using only the limited (20%) human-written gold corrections as training targets.

**Delta**: F1 improved from 14.4 to 16.2 on validation set; perplexity reduced from 10.6 to 9.1.
**Condition**: When only a small fraction of turns have gold correction annotations.

**Evidence**: "Gold corrections provide a strong learning signal."

## [NEGATIVE] Training on Free-form Textual Feedback from 20% Data
Fine-tuning the dialogue model using the free-form textual feedback (the human's response after a bad bot reply) as the training target.

**Delta**: F1 decreased from 14.4 to 13.1 on validation set; perplexity increased from 10.6 to 10.4.
**Condition**: When free-form textual feedback is used directly as a target without correction or filtering.

**Evidence**: "Free-form textual feedback from 20% performed worse than the baseline BB2 3B (F1=13.1 vs. 14.4)."
