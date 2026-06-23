# GPU Memory Management and Device Fallback Patterns

## Memory Check Helper

```python
import torch

def get_safe_device(required_free_bytes: int = 8 * 1024**3) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < required_free_bytes:
        return torch.device('cpu')
    return torch.device('cuda')
```

## Suggested thresholds by model size

| Model                | Min free for training | Min free for inference |
|----------------------|-----------------------|------------------------|
| DeBERTa-v3-base      | 4 GiB                 | 2 GiB                  |
| DeBERTa-v3-large     | 8 GiB                 | 4 GiB                  |

## OOM Recovery During Training

```python
try:
    train_loop(model, dataloader, device)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    model = model.to('cpu')
    dataloader = DataLoader(dataset, batch_size=max(1, batch_size // 4))
    train_loop(model, dataloader, torch.device('cpu'))
```

## Environment Resilience Strategies

1. Clear CUDA cache before model loading.
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. Enable gradient checkpointing during training.
4. Batch-size adaptation: if training OOMs, halve batch size and retry.
5. CPU for inference: frozen-model inference is functionally correct on CPU.
