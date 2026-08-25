from types import SimpleNamespace

import run as run_module


def test_finalize_reserve_preserves_explicit_zero():
    cfg = SimpleNamespace(finalize_reserve_seconds=0)

    assert run_module._resolve_finalize_reserve_seconds(cfg) == 0


def test_finalize_reserve_uses_legacy_default_when_missing_or_none():
    assert run_module._resolve_finalize_reserve_seconds(SimpleNamespace()) == 900
    assert (
        run_module._resolve_finalize_reserve_seconds(
            SimpleNamespace(finalize_reserve_seconds=None)
        )
        == 900
    )


def test_finalize_reserve_clamps_negative_values():
    cfg = SimpleNamespace(finalize_reserve_seconds=-1)

    assert run_module._resolve_finalize_reserve_seconds(cfg) == 0
