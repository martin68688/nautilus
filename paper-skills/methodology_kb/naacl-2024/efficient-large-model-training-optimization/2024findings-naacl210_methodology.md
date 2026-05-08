# Teaching Llama a New Language Through Cross-Lingual Knowledge Transfer

**Source**: https://aclanthology.org/2024.findings-naacl.210/

## [POSITIVE] Continued monolingual pretraining
Additional pretraining of Llama-2-7B on a mixed Estonian-English corpus (75% Estonian, 25% English) using up to 5B tokens from CulturaX.

**Delta**: Performance gains on all Estonian tasks as pretraining size increases (0B → 1B → 5B tokens).
**Condition**: Applied before instruction-tuning; benefits all Estonian tasks (CSR, QA, MT, GEC).

**Evidence**: "We observe performance gains on all Estonian tasks as the size of the pretraining dataset increases."

## [POSITIVE] Cross-lingual instruction-tuning with Alpacas (English + Estonian general task instructions)
Combining Stanford Alpaca (English) with Alpaca-est (Estonian) into a single instruction-tuning dataset.

**Delta**: Baseline model (1) achieves CSR 63.6, QA F1 46.5, MT EN-ET 22.5, MT ET-EN 32.3, GEC 56.6.
**Condition**: Used as baseline instruction-tuning; effective for all tasks.

**Evidence**: "Our results demonstrate that even a relatively small amount of additional monolingual pretraining followed by cross-lingual instruction-tuning significantly enhances results on Estonian."

## [POSITIVE] Sequential training: translation task first, then general task instructions
Training the model first on translation task instructions (TRTASK) and then on general task instructions (Alpacas), rather than combining them.

**Delta**: Without pretraining: QA acc. 64.7 vs 61.2 (combined), MT ET-EN 27.4 vs 1.5 (combined). With 1B pretraining: QA acc. 70.6 vs 63.5 (combined), MT ET-EN 23.0 vs 1.5 (combined).
**Condition**: Most beneficial when no monolingual pretraining is used; with pretraining, benefits diminish except for MT and GEC.

**Evidence**: "We can see that without additional pretraining, the translation task significantly improves the results for QA, machine translation, and GEC."

## [POSITIVE] Supplementing with high-quality English instructions (HQI)
Adding high-quality English instructions (CoT, FLAN-V2, Open Assistant conversations) to the Alpacas dataset.

**Delta**: CSR 66.4 vs 63.6 (Alpacas only), QA F1 52.9 vs 46.5, GEC 59.4 vs 56.6.
**Condition**: Applied to LLAMMAS-BASE; improves all Estonian tasks via cross-lingual transfer.

**Evidence**: "We observe a somewhat surprising increase in all scores when Alpacas is supplemented with high-quality English instructions."

## [POSITIVE] Adding high-quality translation instructions (HQTRTASK) to HQI + Alpacas
Supplementing the HQI + Alpacas mixture with high-quality parallel data from WMT18 dev set and MTee validation set.

**Delta**: QA F1 54.8 vs 52.9 (without HQTRTASK), QA acc. 84 vs 82, GEC 60.3 vs 59.4.
**Condition**: Applied to LLAMMAS-BASE; yields LLAMMAS model.

**Evidence**: "Combining high-quality English instructions with high-quality translation tasks further enhances the knowledge transfer."

## [POSITIVE] Sequential training with HQTRTASK then HQI + Alpacas (LLAMMAS-MT)
First fine-tuning on high-quality translation instructions (HQTRTASK), then on HQI + Alpacas.

**Delta**: MT EN-ET BLEU 26.9 vs 22.6 (LLAMMAS), MT ET-EN BLEU 36.9 vs 34.6, GEC 61.2 vs 60.3.
**Condition**: Best for translation and GEC tasks; slightly lower on CSR and QA compared to LLAMMAS.

**Evidence**: "We observe that the best results for EN→ET, ET→EN, and GEC are obtained with a model that is trained sequentially, with HQTRTASK as the first step of fine-tuning."

## [POSITIVE] Using translation task instructions without monolingual pretraining
Fine-tuning Llama-2-7B (no additional pretraining) on translation task instructions combined with general task instructions.

**Delta**: Without pretraining: QA acc. 64.7 vs 51.8 (no translation), MT EN-ET 24.5 vs 13.9, GEC 51.2 vs 34.2.
**Condition**: Effective when no monolingual pretraining is available; benefits diminish after pretraining.

**Evidence**: "Without additional pretraining, the translation task significantly improves the results for QA, machine translation, and GEC."

## [NEGATIVE] Using translation task instructions after monolingual pretraining
Fine-tuning LLAMMAS-BASE (pretrained on 5B tokens) on translation task instructions before general task instructions.

**Delta**: CSR 62.2 vs 66.4 (without translation step), QA F1 43.5 vs 54.8.
**Condition**: Applied after 5B token pretraining; harms CSR and QA but helps MT and GEC.

**Evidence**: "For QA and commonsense reasoning, omitting the translation task after pretraining tends to produce stronger results compared to models where pretraining is followed by the translation task."

## [NEUTRAL] Balancing translation and general task dataset sizes
Reducing translation task instructions from 1M to 100K to match the size of Alpacas, to avoid imbalance.

**Delta**: 100K translation + Alpacas: CSR 56, QA 71.8, MT EN-ET 21.1, MT ET-EN 1.6, GEC 56.2. Alpacas only: CSR 66, QA 74.1, MT EN-ET 20.8, MT ET-EN 32.4, GEC 50.5.
**Condition**: Applied to LLAMMAS-BASE; does not improve over Alpacas-only baseline.

**Evidence**: "The model does not outperform the Alpacas baseline."

## [POSITIVE] Using both translation directions (EN→ET and ET→EN)
Including 25% or 50% ET→EN translation instructions alongside EN→ET instructions, rather than only EN→ET.

**Delta**: 25% ET→EN: MT ET-EN BLEU 32.6 vs 1.6 (0% ET→EN), GEC 58.1 vs 56.2. 50% ET→EN: CSR 59 vs 56 (0%), QA 76.5 vs 71.8.
**Condition**: Applied to LLAMMAS-BASE with 100K translation instructions; 25% best for MT ET-EN and GEC, 50% best for CSR and QA.

**Evidence**: "For all tasks, having only EN→ET direction is not optimal when translation data is used."

## [NEUTRAL] Using high-quality translation data instead of large low-quality data
Replacing 1M low-quality translation pairs with 6K high-quality pairs from WMT18 and MTee.

**Delta**: 6K high-quality: CSR 57, QA 69.4, MT EN-ET 22.2, MT ET-EN 3.6, GEC 57.5. Alpacas only: CSR 66, QA 74.1, MT EN-ET 20.8, MT ET-EN 32.4, GEC 50.5.
**Condition**: Applied to LLAMMAS-BASE; only GEC benefits, other tasks worse than Alpacas-only.

**Evidence**: "This model also does not outperform the baseline model, except in GEC which seems to benefit from high-quality translation task."

## [POSITIVE] Using chat format for training conversational models
Training models with a conversational format (user/assistant turns) instead of Alpaca-style instruction format, to enable multi-turn dialogue.

**Delta**: Enables multi-turn conversation in Estonian despite no Estonian conversations in training data.
**Condition**: Applied when HQI (containing Open Assistant conversations) is used; enables cross-lingual transfer of conversational ability.

**Evidence**: "Through manual evaluation with 5 conversations, we determine that model (4) (LLAMMAS) can adequately engage in multi-turn conversations."

## [POSITIVE] Loss calculation only on responses
During instruction-tuning, the loss is calculated only on model responses, ignoring user input and instructions.

**Delta**: Standard practice; no quantitative comparison provided.
**Condition**: Applied to all instruction-tuning experiments.

**Evidence**: "During the training, we calculate the loss only on responses, ignoring user input (including multi-turn) and instructions."

## [NEUTRAL] Training for 3 epochs with best checkpoint selection
Models are trained for 3 epochs, and the best checkpoint is selected based on validation loss.

**Delta**: The best checkpoint was always the first checkpoint (trained for 1 epoch).
**Condition**: Applied to all instruction-tuning experiments; indicates overfitting after 1 epoch.

**Evidence**: "We picked the best checkpoint according to the validation loss, which was always the first checkpoint (trained for 1 epoch) in our experiments."

## [POSITIVE] Using diverse prompt templates for translation instructions
Using 32 English and 13 Estonian prompt templates for translation task instructions, following Sanh et al. (2021).

**Delta**: No direct comparison, but cited as important for robustness.
**Condition**: Applied to HQTRTASK dataset creation.

**Evidence**: "Sanh et al. (2021) has demonstrated the importance of using a diverse set of prompts."

## [POSITIVE] Filtering translation bitexts with OpusFilter
Filtering parallel data using long word, sentence length, length-ratio, character score, language-ID, punctuation, and numeral filters.

**Delta**: No direct comparison, but used to create TRTASK dataset.
**Condition**: Applied to all translation task data sources (CCMatrix, WikiMatrix, OpenSubtitles, Europarl).

**Evidence**: "We filter the data with OpusFilter using long word, sentence length, source-target length-ratio, character score, language-ID, terminal punctuation, and non-zero numerals filters."

## [POSITIVE] Using 75% English→Estonian and 25% Estonian→English translation directions
Setting the proportion of translation directions to 75% EN→ET and 25% ET→EN in the TRTASK dataset.

**Delta**: 25% ET→EN gives best MT ET-EN (32.6 BLEU) and GEC (58.1 F0.5) compared to 0% or 50%.
**Condition**: Applied to TRTASK dataset; optimal for translation and GEC tasks.

**Evidence**: "For MTET→EN and GEC 25% ET→EN seems to offer the best scores."

## [NEUTRAL] Using packing for pretraining
Concatenating training examples to fill the model context during continued pretraining.

**Delta**: Standard practice; no quantitative comparison.
**Condition**: Applied to continued pretraining of LLAMMAS-BASE.

**Evidence**: "We use packing for pretraining which means that the training examples are concatenated to fill the model context."
