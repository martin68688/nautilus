from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents import agent_protocol_review_agent
from agents.coder import stepwise_coder
from agents.coder.stepwise_coder import MetaAgent
from agents.memory.experiment_r_router import _call_retrieval_agent
from engine.executor import _candidate_uid_isolation_enabled
from llm import compile_prompt_to_md


def test_meta_agent_accepts_code_only_fenced_completion(monkeypatch):
    monkeypatch.setattr(
        MetaAgent,
        "_build_merge_prompt",
        lambda self, **kwargs: "merge",
    )
    monkeypatch.setattr(
        stepwise_coder,
        "generate",
        lambda **kwargs: "```python\nvalue = 7\n```",
    )
    agent = SimpleNamespace(
        acfg=SimpleNamespace(code=SimpleNamespace(temp=1.0)),
        cfg=SimpleNamespace(),
    )
    plan, code = MetaAgent().merge(
        task_desc="task",
        data_preview_str="preview",
        step_results=[],
        prompt_base={},
        agent_instance=agent,
        context=SimpleNamespace(),
        retries=1,
    )
    assert plan == "Merged the specialist steps into one executable Python pipeline."
    assert "```" not in code
    compile(code, "<test>", "exec")


def test_agent_semantic_review_repairs_then_rechecks_actual_entrypoint(
    tmp_path, monkeypatch
):
    responses = iter(
        [
            {
                "status": "revise",
                "reason": "main reopens the public training table",
                "actual_entrypoint": "main via __main__ guard",
                "findings": ["public train read"],
                "revised_code": (
                    "<<<<<<< SEARCH\n"
                    "    rows = 'public'\n"
                    "=======\n"
                    "    rows = 'host_train_rows'\n"
                    ">>>>>>> REPLACE"
                ),
            },
            {
                "status": "clean",
                "reason": "the actual main path now uses the bound rows",
                "actual_entrypoint": "main via __main__ guard",
                "findings": [],
                "revised_code": "",
            },
        ]
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "query",
        lambda **kwargs: next(responses),
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "get_host_protocol_contract_from_agent",
        lambda _agent: {"task_id": "aerial-cactus-identification"},
    )
    settings = SimpleNamespace(
        agent_semantic_review_enabled=True,
        agent_semantic_max_repair_attempts=2,
        agent_semantic_temperature=0.0,
        agent_semantic_max_tokens=4096,
    )
    agent = SimpleNamespace(
        task_desc="binary image classification",
        acfg=SimpleNamespace(
            protocol_preflight=settings,
            feedback=SimpleNamespace(model="reviewer"),
        ),
        cfg=SimpleNamespace(log_dir=tmp_path),
    )
    node = SimpleNamespace(
        id="node-1",
        code=(
            "def main():\n"
            "    rows = 'public'\n"
            "    return rows\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    )

    source, report = agent_protocol_review_agent.run(agent, node)
    assert "host_train_rows" in source
    assert report["final_status"] == "clean"
    assert report["repairs_applied"] == 1
    assert report["execution_disposition"] == "observe_then_execute"
    assert report["attempts"][0]["patch_applied"] is True
    persisted = json.loads(
        (tmp_path / "agent_semantic_review" / "node-1.json").read_text()
    )
    assert persisted == report


def test_agent_semantic_review_unavailable_is_observed_not_blocked(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "query",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "get_host_protocol_contract_from_agent",
        lambda _agent: {},
    )
    agent = SimpleNamespace(
        task_desc="task",
        acfg=SimpleNamespace(
            protocol_preflight=SimpleNamespace(
                agent_semantic_review_enabled=True,
                agent_semantic_max_repair_attempts=2,
                agent_semantic_temperature=0.0,
                agent_semantic_max_tokens=4096,
            ),
            feedback=SimpleNamespace(model="reviewer"),
        ),
        cfg=SimpleNamespace(log_dir=tmp_path),
    )
    node = SimpleNamespace(id="node-2", code="value = 1\n")
    source, report = agent_protocol_review_agent.run(agent, node)
    assert source == node.code
    assert report["final_status"] == "unavailable"
    assert report["execution_disposition"] == "observe_then_execute"


def test_static_observations_detect_host_sdk_handle_and_evidence_misuse():
    observations = agent_protocol_review_agent._host_sdk_static_observations(
        """
def main():
    session = current_session()
    views = session.get_split()
    rows = list(views.validation)
    with session.prediction_scope(component="model", data_view=views.validation) as prediction_rows:
        predictions = [0.5 for _ in prediction_rows]
        session.evaluate_internal(views.validation, predictions, label_key="label")
"""
    )
    assert {row["code"] for row in observations} == {
        "data_view_handle_direct_use",
        "evaluate_inside_prediction_scope",
    }


def test_static_observations_detect_none_passed_to_iterated_helper_parameter():
    observations = agent_protocol_review_agent._host_sdk_static_observations(
        """
def train_model(train_loader, val_loader):
    for images, labels in train_loader:
        pass
    for images, labels in val_loader:
        pass

def main():
    train_model(train_loader=object(), val_loader=None)
"""
    )
    assert observations == [
        {
            "code": "none_passed_to_iterated_parameter",
            "line": 9,
            "message": (
                "train_model iterates parameter 'val_loader', but this call "
                "passes None; repair the helper/call data flow before execution."
            ),
        }
    ]


def test_agent_semantic_review_retries_false_clean_static_finding(
    tmp_path, monkeypatch
):
    responses = iter(
        [
            {
                "status": "clean",
                "reason": "incorrectly claimed clean",
                "actual_entrypoint": "main",
                "findings": [],
                "revised_code": "",
            },
            {
                "status": "revise",
                "reason": "consume validation rows through prediction_scope",
                "actual_entrypoint": "main",
                "findings": ["DataViewHandle is not iterable"],
                "revised_code": (
                    "<<<<<<< SEARCH\n"
                    "    rows = list(views.validation)\n"
                    "=======\n"
                    "    with session.prediction_scope(component=\"model\", data_view=views.validation) as rows:\n"
                    "        rows = list(rows)\n"
                    ">>>>>>> REPLACE"
                ),
            },
            {
                "status": "clean",
                "reason": "scope-yielded rows are used",
                "actual_entrypoint": "main",
                "findings": [],
                "revised_code": "",
            },
        ]
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "query",
        lambda **kwargs: next(responses),
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "get_host_protocol_contract_from_agent",
        lambda _agent: {},
    )
    settings = SimpleNamespace(
        agent_semantic_review_enabled=True,
        agent_semantic_max_repair_attempts=2,
        agent_semantic_max_review_attempts=5,
        agent_semantic_temperature=0.0,
        agent_semantic_max_tokens=4096,
    )
    agent = SimpleNamespace(
        task_desc="task",
        acfg=SimpleNamespace(
            protocol_preflight=settings,
            feedback=SimpleNamespace(model="reviewer"),
        ),
        cfg=SimpleNamespace(log_dir=tmp_path),
    )
    node = SimpleNamespace(
        id="node-static",
        code=(
            "def main():\n"
            "    session = current_session()\n"
            "    views = session.get_split()\n"
            "    rows = list(views.validation)\n"
            "    return rows\n"
        ),
    )
    source, report = agent_protocol_review_agent.run(agent, node)
    assert "list(views.validation)" not in source
    assert report["final_status"] == "clean"
    assert report["repairs_applied"] == 1
    assert report["attempts"][0]["review_retried"] is True
    assert len(report["attempts"]) == 3


def test_agent_semantic_review_retries_uncertain_before_execution(
    tmp_path, monkeypatch
):
    responses = iter(
        [
            {
                "status": "uncertain",
                "reason": "",
                "actual_entrypoint": "",
                "findings": [],
                "revised_code": "",
            },
            {
                "status": "clean",
                "reason": "second review proved the path",
                "actual_entrypoint": "main",
                "findings": [],
                "revised_code": "",
            },
        ]
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "query",
        lambda **kwargs: next(responses),
    )
    monkeypatch.setattr(
        agent_protocol_review_agent,
        "get_host_protocol_contract_from_agent",
        lambda _agent: {},
    )
    agent = SimpleNamespace(
        task_desc="task",
        acfg=SimpleNamespace(
            protocol_preflight=SimpleNamespace(
                agent_semantic_review_enabled=True,
                agent_semantic_max_repair_attempts=2,
                agent_semantic_max_review_attempts=4,
                agent_semantic_temperature=0.0,
                agent_semantic_max_tokens=4096,
            ),
            feedback=SimpleNamespace(model="reviewer"),
        ),
        cfg=SimpleNamespace(log_dir=tmp_path),
    )
    source, report = agent_protocol_review_agent.run(
        agent, SimpleNamespace(id="node-uncertain", code="def main():\n    return 1\n")
    )
    assert source == "def main():\n    return 1\n"
    assert report["final_status"] == "clean"
    assert report["attempts"][0]["review_retried"] is True
    assert len(report["attempts"]) == 2


def test_real_retrieval_agent_prompt_compiles_structured_observations(
    monkeypatch,
):
    captured = {}

    def fake_query(**kwargs):
        captured["compiled"] = compile_prompt_to_md(kwargs["system_message"])
        return {"action": "finish", "reason": "done", "selected_ids": []}

    import llm

    monkeypatch.setattr(llm, "query", fake_query)
    layer = SimpleNamespace(
        cfg=SimpleNamespace(
            agent=SimpleNamespace(
                feedback=SimpleNamespace(model="reviewer"),
                code=SimpleNamespace(model="solver"),
            )
        ),
        experiment_r_agentic_temperature=0.0,
        experiment_r_agentic_max_tokens=1200,
        nodes={"n1": {"task": "task", "text": "clean candidate"}},
    )
    action = _call_retrieval_agent(
        layer,
        stage="draft",
        task_id="task",
        task_desc="description",
        query_text="query",
        trace=[{"observation": {"tool": "search", "candidate_ids": ["n1"]}}],
        known={"n1": {"id": "n1", "source": "runforest", "score": 1.0}},
    )
    assert action["action"] == "finish"
    assert '"candidate_ids"' in captured["compiled"]


def test_shadow_host_can_retain_uid_isolation_without_enforcing_receipts():
    cfg = SimpleNamespace(
        evaluation_authority=SimpleNamespace(
            protocol_runtime_mode="host_sdk_shadow"
        ),
        agent=SimpleNamespace(
            protocol_preflight=SimpleNamespace(candidate_process_isolation=True)
        ),
    )
    assert _candidate_uid_isolation_enabled(cfg) is True
    cfg.agent.protocol_preflight.candidate_process_isolation = False
    assert _candidate_uid_isolation_enabled(cfg) is False
