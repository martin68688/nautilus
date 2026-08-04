import hashlib
import json
import pickle
import random


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def capture_rng_state() -> dict[str, str]:
    """Commit to live local RNG states without claiming provider determinism."""

    components: dict[str, str] = {
        "python": _digest(pickle.dumps(random.getstate(), protocol=4)),
    }
    try:
        import numpy as np  # type: ignore

        state = np.random.get_state()
        payload = json.dumps(
            [
                state[0],
                state[1].tolist(),
                int(state[2]),
                int(state[3]),
                float(state[4]),
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        components["numpy"] = _digest(payload)
    except Exception:
        components["numpy"] = "unavailable"
    try:
        import torch  # type: ignore

        components["torch_cpu"] = _digest(
            torch.get_rng_state().cpu().numpy().tobytes()
        )
        if torch.cuda.is_available():
            cuda = b"".join(
                state.cpu().numpy().tobytes()
                for state in torch.cuda.get_rng_state_all()
            )
            components["torch_cuda"] = _digest(cuda)
        else:
            components["torch_cuda"] = "unavailable"
    except Exception:
        components["torch_cpu"] = "unavailable"
        components["torch_cuda"] = "unavailable"
    canonical = json.dumps(
        components, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"rng_state_hash": _digest(canonical), **components}


def set_global_seed(seed: int) -> dict[str, str]:
    """Seed local RNGs and return a hash commitment to their live states."""

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return capture_rng_state()

