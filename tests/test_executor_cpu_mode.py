from __future__ import annotations

from types import SimpleNamespace

from engine.executor import Interpreter, _candidate_uid_isolation_enabled


def _cfg(num_gpus: int) -> SimpleNamespace:
    return SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        agent=SimpleNamespace(
            search=SimpleNamespace(
                parallel_search_num=1,
                num_gpus=num_gpus,
            )
        ),
        evaluation_authority=SimpleNamespace(
            mode="off",
            runtime_protocol_observer_enabled=False,
        ),
    )


def test_cpu_mode_runs_without_gpu_modulo_and_hides_cuda(tmp_path):
    result = Interpreter(tmp_path, timeout=10, cfg=_cfg(0)).run(
        "import os\nprint('CUDA=' + repr(os.environ.get('CUDA_VISIBLE_DEVICES')))\n",
        "cpu-only-candidate",
    )

    assert result.exc_type is None
    assert "CUDA=''" in "".join(result.term_out)


def test_negative_gpu_count_is_rejected(tmp_path):
    result = Interpreter(tmp_path, timeout=10, cfg=_cfg(-1)).run(
        "print('must not run')\n",
        "invalid-gpu-count",
    )

    assert result.exc_type == "RuntimeError"
    assert "agent.search.num_gpus must be non-negative" in "".join(
        result.term_out
    )


def test_disabled_preflight_does_not_drop_candidate_uid():
    cfg = _cfg(0)
    cfg.agent.protocol_preflight = SimpleNamespace(
        enabled=False,
        candidate_process_isolation=True,
    )

    assert _candidate_uid_isolation_enabled(cfg) is False


def test_enabled_preflight_keeps_candidate_uid_isolation():
    cfg = _cfg(0)
    cfg.agent.protocol_preflight = SimpleNamespace(
        enabled=True,
        candidate_process_isolation=True,
    )

    assert _candidate_uid_isolation_enabled(cfg) is True
