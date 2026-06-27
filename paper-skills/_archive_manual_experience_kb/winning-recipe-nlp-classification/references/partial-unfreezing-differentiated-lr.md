# Partial Unfreezing + Differentiated Learning Rates

## Finding
DeBERTa-v3-large partial unfreezing (last 8/24 layers) with backbone lr=2e-5 and head lr=5e-5 is the optimal fine-tuning strategy for small datasets.

## Evidence
- Run8 top1: partial unfreezing (last 8 layers) + differentiated LR → val=0.0725
- Run8 Branch2: full fine-tuning → ceiling at ~0.26 (8 variants, all 0.26~0.34)
- Run8 top17: fully frozen backbone → val=0.3853
- Run 091845 top4: same strategy in MSD model → val=0.2517 (ensemble)

## Mechanism
- Freezing first 16 layers preserves pretrained linguistic knowledge
- Only top 8 layers adapt to task-specific features
- Head LR 2.5x higher allows randomly-initialized classifier to converge quickly
- Without differentiated LR, large gradients from untrained head destabilize backbone

## Condition
Training samples < 50K, model > 100M parameters, pretrained Transformer.
