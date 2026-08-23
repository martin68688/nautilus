from __future__ import annotations

import time
from types import SimpleNamespace

from engine.executor import Interpreter
from engine.search_node import SearchNode


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        start_cpu_id=0,
        cpu_number=3,
        agent=SimpleNamespace(
            search=SimpleNamespace(
                parallel_search_num=3,
                num_gpus=3,
            ),
        ),
        evaluation_authority=SimpleNamespace(
            mode="off",
            runtime_protocol_observer_enabled=False,
        ),
    )


def _metric(value: float):
    return SimpleNamespace(value=value, maximize=False)


def test_preferred_process_id_pins_candidate_to_matching_gpu(tmp_path):
    result = Interpreter(tmp_path, timeout=10, cfg=_cfg()).run(
        "import os\nprint('CUDA=' + repr(os.environ.get('CUDA_VISIBLE_DEVICES')))\n",
        "dedicated-lane-candidate",
        preferred_process_id=2,
    )

    assert result.exc_type is None
    assert "CUDA='2'" in "".join(result.term_out)


def test_preferred_process_id_must_address_allocated_slot(tmp_path):
    interpreter = Interpreter(tmp_path, timeout=10, cfg=_cfg())

    try:
        interpreter.run(
            "print('must not run')\n",
            "invalid-dedicated-lane",
            preferred_process_id=3,
        )
    except ValueError as error:
        assert "preferred_process_id" in str(error)
    else:
        raise AssertionError("out-of-range dedicated lane was accepted")


def test_each_executor_slot_has_its_own_candidate_deadline(tmp_path):
    interpreter = Interpreter(tmp_path, timeout=100, cfg=_cfg())
    now = time.monotonic()
    interpreter.set_run_deadline(now + 100, finalize_reserve_seconds=10)
    interpreter.set_slot_deadline(2, now + 20)

    assert 0 < interpreter.remaining_work_seconds(2) <= 20
    assert interpreter.remaining_work_seconds(0) > 80


def test_required_branch_selection_cannot_escape_to_global_best(monkeypatch):
    from engine import node_selection

    virtual_root = SearchNode(id="root", code="", plan="root", stage="root")
    transfer = SearchNode(
        id="transfer",
        code="print('transfer')",
        plan="transfer",
        parent=virtual_root,
        stage="draft",
        draft_role="memory_transfer",
        branch_id=7,
        is_buggy=False,
        is_valid=True,
        metric=_metric(0.9),
    )
    novel = SearchNode(
        id="novel",
        code="print('novel')",
        plan="novel",
        parent=virtual_root,
        stage="draft",
        draft_role="novel_exploration",
        branch_id=8,
        is_buggy=False,
        is_valid=True,
        metric=_metric(0.1),
    )
    calls = []

    def _select(_agent, root, branch_id):
        calls.append((root.id, branch_id))
        return root

    monkeypatch.setattr(node_selection, "_select_within_branch", _select)
    agent = SimpleNamespace(
        virtual_root=virtual_root,
        branch_all_nodes={7: [transfer], 8: [novel]},
        branch_successful_nodes={7: [transfer], 8: [novel]},
        _focused_branch_root_id=None,
        _focused_branch_lock_reason=None,
        _focused_branch_locked_at_step=None,
        journal=[virtual_root, transfer, novel],
    )

    selected = node_selection.select_from_branch_id(agent, 7)

    assert selected is transfer
    assert calls == [("transfer", 7)]
