# Plug-in Language Model: Controlling Text Generation with a Simple Regression Model

**Source**: https://aclanthology.org/2024.findings-naacl.139/

## [POSITIVE] Reinforcement Learning (RL) for Latent State Adjustment
Using reinforcement learning to adjust the latent state (key-value pairs) of a pre-trained language model during inference, with rewards from black-box tools evaluating future generated sequences.

**Delta**: Outperforms existing state-of-the-art methods (gradient-based, weighted decoding, prompt-based) across all metrics.
**Condition**: Applicable when controlling text generation with specific attributes (sentiment, topic, detoxification) without fine-tuning the base model.

**Evidence**: "PiLM leverages reinforcement learning to utilize black box tools directly, adjusting the latent state to control text generation."

## [POSITIVE] Latent Controller (Regression Model)
A simple 2-layer fully connected neural network that predicts the modified latent state from the unmodified one, replacing the time-consuming backpropagation in RL.

**Delta**: Inference time reduced from 21.39s (PiLM-RL) to 2.2s (PiLM-Controller), comparable to GPT-2's 1.39s.
**Condition**: Used during inference to accelerate controlled generation; requires training pairs collected from RL updates.

**Evidence**: "By replacing backpropagation with a simple regression model, PiLM can achieve an inference time comparable to that of the original LLM."

## [POSITIVE] Future Sequence Sampling for Reward
Sampling future n tokens and using black-box tools to evaluate the entire sequence, rather than a single token, to compute the reward for latent state updates.

**Delta**: n=10 yields perplexity 12.83 vs n=1 yields 124.76; longer sequences improve quality but reduce control strength if update count decreases.
**Condition**: Applied during RL updates; n is a hyperparameter (e.g., 10 in experiments).

**Evidence**: "Considering longer future tokens allows PiLM to use more information to guide the language model."

## [POSITIVE] Black-Box Tool Integration
Using pre-existing black-box attribute models (e.g., sentiment classifiers, toxicity detectors) as reward functions without needing access to the language model's latent states.

**Delta**: Enables use of tools like Perspective API and HuggingFace classifiers; outperforms gradient-based methods that require latent states.
**Condition**: Applicable when attribute classifiers or tools only accept text as input (e.g., sentiment analysis, toxicity detection).

**Evidence**: "PiLM leverages reinforcement learning to utilize black box tools directly, adjusting the latent state to control text generation."

## [POSITIVE] Selective Latent Update (Every n Tokens)
Updating the latent representation H only every n tokens instead of at every position, to avoid excessive updates and maintain text quality.

**Delta**: Outperforms baselines in fluency (perplexity 13.71 vs 263.53 for FUDGE) and control strength.
**Condition**: Used in both PiLM-RL and PiLM-Controller; n is a hyperparameter (e.g., 10).

**Evidence**: "Deviating from previous approaches... adjusting the distribution at all positions may result in excessive updates, resulting in a decline in text quality."

## [POSITIVE] Lemmatization for Topic Word Matching
Applying lemmatization to reduce inflected words to their canonical form before matching against topic word lists, improving recall of related words.

**Delta**: Held-out bag hit rate 0.97 vs PPLM's 0.69; topic coverage 92.86% vs PPLM's 89.05%.
**Condition**: Used in topic control task to evaluate generated text against topic word lists.

**Evidence**: "We introduce lemmatization to prevent overly strict comparisons that could result in inaccuracies in the scoring calculations."

## [NEUTRAL] Dynamic M and Reset Hidden
Dynamic M determines whether to use RL to adjust hidden state based on reward score; Reset hidden recomputes all hidden states before update.

**Delta**: Enhances text quality but reduces control strength (trade-off).
**Condition**: Optional techniques used in PiLM-RL; applicable when quality is prioritized over control strength.

**Evidence**: "Both of these techniques can enhance text quality, but they come with a trade-off as they tend to reduce control strength."

## [POSITIVE] Perplexity Weight in Reward Function
Including a perplexity term (with negative weight) in the reward function to penalize fluency degradation during RL updates.

**Delta**: PiLM achieves perplexity 13.71 (close to GPT-2's 11.05) vs FUDGE's 263.53.
**Condition**: Used in sentiment control task; w_ppl hyperparameter ranges from -0.2 to -0.05.

**Evidence**: "The plug-in module also considers the perplexity derived from G to improve fluency."

## [NEUTRAL] Top-k Sampling with k=10
Using top-k sampling with k=10 during generation to balance diversity and control.

**Delta**: Not explicitly compared, but used consistently across all experiments.
**Condition**: Standard decoding strategy used in all tasks for comparability with baselines.

**Evidence**: "Generate three sequences with a length of 50 tokens using top-k sampling with k = 10."
