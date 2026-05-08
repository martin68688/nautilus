# Start Simple: Progressive Difficulty Multitask Learning

**Source**: https://aclanthology.org/2024.naacl-srw.7/

## [POSITIVE] Progressive Difficulty Multitask Learning
Training a neural network simultaneously on subtasks of increasing difficulty (e.g., coarse to fine label granularity) to improve performance on the main task.

**Delta**: outperforms baseline across all tasks
**Condition**: Applied to sentiment analysis, text classification, unit segmentation, and syllogistic reasoning; works with FCNN, CNN, LSTM, Longformer, and GPT.

**Evidence**: "We found that by training the model on both the main task and its simplified versions simultaneously, the performance of the model improved in all cases."

## [POSITIVE] Word2Vec Pretrained Embeddings (Frozen)
Using word2vec skip-gram to pretrain 32-dimensional word embeddings on each dataset, then freezing them during backbone model training.

**Delta**: descriptive: 'better generalization'
**Condition**: Used in non-transformer-based models (FCNN, CNN, LSTM) for sentiment analysis and text classification.

**Evidence**: "According to Mikolov et al. (2013b) and Levy et al. (2015), a pretraining strategy like this initializes the model with semantically meaningful representations, which can capture the contextual and syntactic similarities between words and result in better generalization."

## [POSITIVE] Batch Normalization
Applying batch normalization to every layer except the embedding layer and output layers to accelerate convergence.

**Delta**: descriptive: 'accelerate convergence'
**Condition**: Applied to all models in experiment 1 (FCNN, CNN, LSTM, Longformer).

**Evidence**: "In order to accelerate convergence, we applied batch normalization (Ioffe and Szegedy, 2015) to every layer in the model except for the embedding layer and the output layers."

## [POSITIVE] Deep CNN (17 convolutional layers)
Stacking 17 convolutional layers with max pooling and fully connected layers, inspired by Conneau et al. (2017), to increase feature extractor depth.

**Delta**: e.g., Amazon L3 accuracy: 26.77% (shallow) vs 34.13% (deep)
**Condition**: Applied to sentiment analysis and text classification; improves performance for CNN but not for LSTM.

**Evidence**: "We observed that while trained using progressive difficulty MTL, the CNN may be benefited from stacking more layers."

## [NEGATIVE] Deep LSTM (8 layers with residual connections)
Stacking 8 LSTM layers with residual connections between every two layers in the middle six layers, inspired by Kim et al. (2017).

**Delta**: e.g., Coronavirus tweets L2: 53.55% (shallow) vs 39.57% (deep)
**Condition**: Applied to sentiment analysis and text classification; degrades performance compared to shallow LSTM.

**Evidence**: "The LSTM does not improve as much. ... We attribute this difference to the distinction in their field of view, since CNNs are structured so that each layer captures increasingly complex features, whereas LSTMs have an architectural bottleneck."

## [NEGATIVE] Transfer Learning with Complementary Datasets
Adding named entity recognition (CoNLL-2003) and text classification (DBPedia) as auxiliary tasks to unit segmentation via progressive difficulty MTL.

**Delta**: L3 accuracy: 70.58% (no TL) vs 70.57% (CoNLL-2003) vs 70.53% (DBPedia)
**Condition**: Applied to unit segmentation with Longformer; no positive effect observed.

**Evidence**: "Table 4 indicates that TL with complementary data sets cannot improve the performance of MTL with progressive difficulty subtasks."

## [POSITIVE] Progressive Chain-of-Thought (CoT) Prompting
Prompting the LLM to first summarize relations and perform basic inference before answering the main syllogistic reasoning question, instead of a direct true/false answer.

**Delta**: +5.17% accuracy (72.99% baseline vs 78.16% progressive CoT)
**Condition**: Applied to syllogistic reasoning using GPT 3.5.

**Evidence**: "Progressive CoT achieved notable improvement, without an extensive prompt design."

## [NEUTRAL] Weighted Sum Loss for Multitask Learning
Combining losses from each subtask using a weighted sum, where each subtask uses a softmax-loss function.

**Delta**: descriptive: 'used as part of the MTL framework'
**Condition**: Applied to all models in experiment 1.

**Evidence**: "The loss is a weighted sum of the loss of each subtask, and the loss function in every subtask is a softmax-loss function."

## [POSITIVE] Longformer for Long-Document Unit Segmentation
Using Longformer instead of BERT to handle long input sequences (up to 1024 tokens) with improved computational efficiency via dilated sliding window attention.

**Delta**: 70.58% accuracy (statistically significant improvement over baseline)
**Condition**: Applied to unit segmentation on argumentative essays dataset.

**Evidence**: "While BERT ... performance diminishes when applied to lengthy texts. To this end, we use Longformer, which not only performs better with long input sequences but also improves computational efficiency."
