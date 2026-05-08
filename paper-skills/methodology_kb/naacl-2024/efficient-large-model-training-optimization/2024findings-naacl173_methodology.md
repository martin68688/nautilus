# Compensate Quantization Errors: Make Weights Hierarchical to Compensate Each Other

**Source**: https://aclanthology.org/2024.findings-naacl.173/

## [POSITIVE] Learnable Singular Value Increment (LSI)
Introduces a learnable increment added to the singular values of weight matrices via SVD, allowing slight perturbations to the weight distribution to help weights compensate for quantization errors globally.

**Delta**: Outperforms baselines (GPTQ, AWQ, OmniQuant) across multiple settings; e.g., OPT-125M W2A16-g128 perplexity drops from 75.43 (OmniQuant) to 56.17.
**Condition**: Effective for weight-only and weight-activation quantization, especially for smaller models and low-bit settings (e.g., W2, W3). Benefits diminish with group-wise scaling in large models.

**Evidence**: "LSI provides effective and powerful solutions for the quantization of smaller LLMs and w2 settings while helping further improve the performance in more sophisticated settings."

## [POSITIVE] Additional Square Matrix for Group-Wise Scaling
A small square matrix (e.g., 100x100 to 600x600) added to the top-left of the diagonal singular value matrix to help balance quantization across groups in group-wise settings.

**Delta**: Improves perplexity in group-wise settings; e.g., OPT-125M W3A16g128 perplexity improves from 32.25 (OmniQuant) to 31.06 (Ours k100).
**Condition**: Applied when group-wise scaling is used (e.g., g128, g64). Benefits are more pronounced for smaller models; for larger models, benefits diminish and may cause overfitting.

**Evidence**: "To address this issue, we introduced an additional small square matrix... helps achieve a better balance in the group-wise setting."

## [POSITIVE] Integration with Smooth and Clipping Techniques (LET + LWC)
Combines LSI with Learnable Equivalent Transformation (LET) and Learnable Weight Clipping (LWC) from OmniQuant to transfer quantization difficulty and handle outliers.

**Delta**: Significantly outperforms OmniQuant in W4A4 settings; e.g., OPT-6.7B W4A4 WikiText2 perplexity drops from 12.24 (OmniQuant) to 11.82.
**Condition**: Applied in both weight-only and weight-activation quantization settings. Essential for achieving state-of-the-art results, especially in low-bit weight-activation quantization.

**Evidence**: "By incorporating these techniques with recent advancements... LSI can significantly reduce quantization errors and achieve remarkable performance gains."

## [POSITIVE] LSI for Finetuning Quantized Models
Using LSI only on the last few layers of a quantized model to quickly finetune on a specific dataset, leveraging the overfitting tendency to improve performance on that dataset.

**Delta**: LLaMA-7B W3A16g128 PTB perplexity improves from 33.45 (OmniQuant) to 30.69 (Omni w/ LSI).
**Condition**: Effective for few-shot finetuning on specific datasets (e.g., PTB). Works with various baseline quantization methods (RTN, OmniQuant, GPTQ).

**Evidence**: "Employing LSI only on the last several layers of a LLM can also result in improved performance on a specific dataset without largely compromising other abilities."

## [POSITIVE] Low Learning Rate and AdamW Optimizer
Training LSI parameters with a low learning rate (2e-4) and AdamW optimizer with weight decay 0 to avoid large distribution shifts.

**Delta**: Enables stable training; larger models require fewer epochs (e.g., OPT-30B only 2 epochs on 32 samples).
**Condition**: Used during LSI training for all quantization settings. Necessary to prevent instability and overfitting.

**Evidence**: "Singular values can introduce significant variations in the distribution of weights, so we maintain a low learning rate at 2e-4."

## [POSITIVE] Data-Free Post-Training Quantization (PTQ) with Small Calibration Set
LSI is trained using only a small calibration set (e.g., 32-128 samples from WikiText2) without requiring full training data.

**Delta**: Achieves state-of-the-art results with minimal data; e.g., OPT-30B W4A16g128 trained on only 32 samples.
**Condition**: Applied in all PTQ experiments. Works for both weight-only and weight-activation quantization.

**Evidence**: "All the data used in our training was collected from WikiText2. Notably, the training process is quite fast, with larger models requiring fewer epochs."

## [NEGATIVE] LSI-Only Quantization (Without Smoothing)
Using LSI alone without smoothing techniques to adjust weight distribution for quantization.

**Delta**: Suffers from severe overfitting; e.g., LLaMA-7B W3A16g128 WikiText2 perplexity 6.25 vs GPTQ 6.55, but C4 perplexity 7.91 vs GPTQ 7.85 (worse).
**Condition**: When used without smoothing techniques (LET/LWC). Not recommended for general quantization; only useful for targeted finetuning.

**Evidence**: "LSI alone has a grave problem of overfitting... LSI obtains relatively good performance through significant overfitting on a specific dataset."

## [NEUTRAL] Random Initialization of LSI Parameters
Initializing LSI parameters randomly instead of using pre-trained OmniQuant parameters.

**Delta**: Potentially better results but unstable and requires longer training.
**Condition**: Alternative initialization strategy. Not used in main experiments due to instability and longer training time.

**Evidence**: "Our best results were not obtained by using OmniQuant’s parameters as the initialization... random initialization followed by a longer training period could potentially yield better results."

## [POSITIVE] Training Only Last Few Layers for Finetuning
Applying LSI only to the last two layers of the model during finetuning to improve efficiency and reduce overfitting on other tasks.

**Delta**: LLaMA-30B W4A16 PTB perplexity improves from 16.48 (OmniQuant) to 16.46 (Omni w/ LSI) with only last two layers trained.
**Condition**: Used for finetuning quantized models on specific datasets. Reduces computational cost and preserves generalization.

**Evidence**: "Employing LSI only on the last several layers of a LLM can also result in improved performance on a specific dataset without largely compromising other abilities."

## [POSITIVE] Keeping Softmax in Float32
Maintaining the softmax operation in full precision (float32) during inference to avoid excessive disturbance from quantization in self-attention layers.

**Delta**: Helps maintain performance in weight-activation quantization settings.
**Condition**: Applied in all weight-activation quantization experiments (e.g., W6A6, W4A4).

**Evidence**: "We adhere to the original setup, keeping the Softmax part in float32, as this helps mitigate excessive disturbance caused by self-attention layers during the inference process."
