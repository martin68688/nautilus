# Resource and Memory Management

## Table of Contents
1. [GPU Memory Pre-check](#gpu-memory-pre-check)
2. [Memory-Safe Training Configuration](#memory-safe-training-configuration)
3. [OOM Fallback Script](#oom-fallback-script)
4. [Gradient Checkpointing vs Accumulation](#gradient-checkpointing-vs-accumulation)
5. [Disk and Shared Memory Checks](#disk-and-shared-memory-checks)
6. [DataLoader Settings](#dataloader-settings)
7. [Incremental Pipeline Scaling](#incremental-pipeline-scaling)

## GPU Memory Pre-check

```python
import torch

def check_gpu_memory(min_free_gb=8.0):
    if not torch.cuda.is_available():
        return False
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_gb:
        print(f"WARNING: Only {free_gb:.1f} GB free")
        return False
    return True
```

**Key rule:** Always use the *free* memory figure, not total capacity.

## Memory-Safe Training Configuration

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# AMP with gradient accumulation
scaler = torch.cuda.amp.GradScaler()
accumulation_steps = 4
```

### VRAM Sizing Table

| Model | Batch Size | Max Seq Len | Approx VRAM |
|-------|-----------|-------------|-------------|
| DeBERTa-v3-base  | 16 | 256 | ~6–8 GiB  |
| DeBERTa-v3-base  | 16 | 512 | ~10–14 GiB |
| DeBERTa-v3-large | 8  | 256 | ~10–14 GiB |
| DeBERTa-v3-large | 16 | 512 | ~30–40 GiB |

## OOM Fallback Script

```python
def train_with_oom_fallback(model_name, train_fn, dataloader_factory):
    batch_sizes = [16, 8, 4, 2]
    model_fallbacks = [model_name, model_name.replace('large', 'base')]
    for model_attempt in model_fallbacks:
        for bs in batch_sizes:
            try:
                torch.cuda.empty_cache()
                return train_fn(model_attempt, dataloader_factory(batch_size=bs))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
    raise RuntimeError("All configurations resulted in OOM")
```

## Gradient Checkpointing vs Accumulation

| Technique | Can combine with AMP? | Can combine with accumulation? | Can combine with checkpointing? |
|---|---|---|---|
| Mixed precision (AMP) | — | ✅ Yes | ✅ Yes |
| Gradient accumulation | ✅ Yes | — | ❌ **NO** |
| Gradient checkpointing | ✅ Yes | ❌ **NO** | — |

## Disk and Shared Memory Checks

```python
import shutil, os
total, used, free = shutil.disk_usage("/")
print(f"Disk free: {free / (1024**3):.1f} GB")

if os.path.exists("/dev/shm"):
    _, _, shm_free = shutil.disk_usage("/dev/shm")
    print(f"Shared memory free: {shm_free / (1024**3):.1f} GB")
```

## DataLoader Settings

```python
DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=False, drop_last=True)
```

## Incremental Pipeline Scaling

### Phase 1: Baseline
- Single transformer (base variant), conservative batch size, 1–2 epochs.

### Phase 2: Scale up
- Increase sequence length or batch size if memory allows.

### Phase 3: Augment
- Add auxiliary features and ensemble components only after baseline validated.

**DO NOT** load multiple large models simultaneously in a first attempt.