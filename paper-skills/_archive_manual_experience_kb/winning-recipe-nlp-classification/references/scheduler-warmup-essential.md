# Scheduler with Warmup is Essential, Exact Type is Secondary

## Finding
10% warmup phase is critical. Choice between CosineWarmRestarts and LinearDecay matters less.

## Evidence
- Run8 top1: CosineWarmRestarts + warmup → val=0.0725
- Run 091845 top4: LinearDecay + warmup → val=0.2517 (ensemble)
- Run8 top2: NO scheduler → val=0.1457 (2x worse, same branch)

## Mechanism
- Warmup: LR 0→target over 10% steps, prevents destroying pretrained weights
- Decay: cosine restarts help escape local optima; linear decay enables fine convergence
- Both work; having warmup is the critical factor

## Condition
Fine-tuning pretrained Transformers. Warmup ratio 0.05-0.15. Never train without scheduler.
