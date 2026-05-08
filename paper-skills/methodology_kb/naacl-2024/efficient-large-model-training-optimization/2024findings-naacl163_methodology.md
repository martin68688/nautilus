# Towards an On-device Agent for Text Rewriting

**Source**: https://aclanthology.org/2024.findings-naacl.163/

## [POSITIVE] Synthetic Paired Dataset from Continued Generations
Generating diverse, short-form, message-like training data by using few-shot prompting of an LLM to continue generating examples from a small seed query set.

**Delta**: outperforms baselines
**Condition**: Used for supervised fine-tuning (SFT) data generation, especially for in-domain message-like data.

**Evidence**: "The continued generations enable efficient generation of diverse paired data."

## [POSITIVE] LLM-guided Data Selection with CoT and Self-Consistency
Using an off-the-shelf LLM with few-shot Chain-of-Thought reasoning and self-consistency to critique and filter synthetic data, keeping only samples approved by all judges.

**Delta**: outperforms baselines
**Condition**: Applied during data filtering stage to improve quality of synthetic instruction tuning dataset.

**Evidence**: "We propose to use LLMs to critique the generated data. We leverage the few-shot Chain-of-Thoughts (CoT) reasoning of the off-the-shelf LLM to judge whether the response is following the instruction... we only keep the data when it is approved by all LLM judges."

## [POSITIVE] Heuristic-based Reinforcement Learning (Heuristic RL)
Replacing a trained reward model with a weighted combination of heuristic signals (NLI, reversed NLI, length ratio, edit distance ratio, n-gram frequency) as the reward for PPO reinforcement learning.

**Delta**: +1.11 SARI, +2.72 BLEU, +3.13 Update-R on MESSAGEREWRITEEVAL (SFT vs SFT+heuristic RL)
**Condition**: Applied after supervised fine-tuning to further align the model for text rewriting tasks.

**Evidence**: "Heuristic reinforcement learning generally boosts the model's performance on all tasks."

## [POSITIVE] Critique Distillation via Suffix-based Discriminative Fine-tuning
Distilling the critiquing ability of a larger LLM into the on-device model by appending a 'good'/'bad' suffix to the model's response, then fine-tuning the model to generate this suffix, enabling self-critique in a single generation step.

**Delta**: +1.26 SARI, +1.93 BLEU, +0.75 Update-R on MESSAGEREWRITEEVAL (RL vs RL+critique distillation)
**Condition**: Applied after RL to enable efficient self-critique for cascading decisions.

**Evidence**: "The model's overall performance is further improved on SARI, BLEU, and Update-R with little regression on Reversed NLI when we combine the distilled discriminative dataset with the generative dataset."

## [POSITIVE] Cascading with Suffix Score Threshold
Using the probability of the 'good' suffix (suffix score) from the on-device model to decide whether to invoke a larger server-side model; if the score is below a threshold, the server model is called.

**Delta**: With 40% server calls, overall performance is quite close to the full server model (InsGPT).
**Condition**: Applied at inference time to balance performance and cost, especially when a server-side model is available.

**Evidence**: "With 40% server calls, the overall performance is already quite close to the full server model."

## [POSITIVE] Suffix Score vs LM Score for Cascading
Comparing the distilled suffix score (probability of 'good') against the standard LM score (likelihood of generated text) as a quality estimate for deciding when to cascade.

**Delta**: Suffix score with 1 sample outperforms LM score with 1 sample by large margin.
**Condition**: Used in the cascading framework to select high-quality outputs and reduce server calls.

**Evidence**: "Suffix score with 1 sample is outperforming LM score with 1 sample by large margin. This indicates that given a text output, suffix score offers higher quality estimates."

## [NEGATIVE] Ablation of Heuristic Rewards
Removing individual heuristic components (NLI s-t, NLI t-s, length ratio, edit distance, n-gram) from the reward function to measure their contribution.

**Delta**: Removing NLI s-t reduces SARI by 0.84, BLEU by 1.69, Update-R by 2.11; removing NLI t-s reduces Reversed NLI by 0.07.
**Condition**: Ablation study on MESSAGEREWRITEEVAL; each heuristic contributes positively to overall performance.

**Evidence**: "Removing any one of the proposed heuristics will reduce the overall quality of rewrites. Notably the NLI s-t and the NLI t-s play more important roles for securing good rewrite comparing to other rewards."
