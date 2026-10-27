from pathlib import Path
import json
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from engine.research_loop import ResearchLedger, ledger_for  # noqa: E402
from engine.search_node import SearchNode  # noqa: E402
from utils.metric import MetricValue  # noqa: E402

DIRS = {"targets": "labels", "features": "inputs", "model": "model"}


def _node(stage, parent=None, score=None, buggy=False, plan=""):
    node = SearchNode(code="c", plan=plan, stage=stage, parent=parent, is_buggy=buggy,
                      metric=None if score is None else MetricValue(score, maximize=True))
    return node


def test_choose_tries_every_direction_then_favours_confirmed_ones():
    ledger = ResearchLedger(DIRS, ucb_c=0.1)
    base = _node("draft", score=0.040)
    for name, end in (("targets", 0.045), ("features", 0.039), ("model", 0.038)):
        assert ledger.choose() == name
        child = _node("improve", parent=base, score=end, plan=f"DIRECTION: {name}\nHYPOTHESIS: try {name}")
        ledger.register(child, name)
        ledger.conclude(child, threshold=1e-4)
    assert ledger.choose() == "targets"
    verdicts = {r["direction"]: r["verdict"] for r in ledger.records.values()}
    assert verdicts == {"targets": "confirmed", "features": "refuted", "model": "refuted"}


def test_failed_attempt_is_concluded_by_its_repair_against_the_start_node():
    ledger = ResearchLedger(DIRS)
    base = _node("draft", score=0.040)
    attempt = _node("improve", parent=base, buggy=True, plan="DIRECTION: model\nHYPOTHESIS: deeper trees help")
    ledger.register(attempt, "targets")  # the plan's own valid direction wins over the schedule
    ledger.conclude(attempt, threshold=1e-4)
    assert ledger.records[attempt.id]["verdict"] == "failed"
    repair = _node("debug", parent=attempt, score=0.043)
    ledger.conclude(repair, threshold=1e-4)
    rec = ledger.records[attempt.id]
    assert rec["direction"] == "model" and rec["verdict"] == "confirmed"
    assert abs(rec["delta"] - 0.003) < 1e-9 and rec["result_node_id"] == repair.id


def test_prompt_shows_direction_format_and_ledger(tmp_path):
    ledger = ResearchLedger(DIRS, path=tmp_path / "research_ledger.json")
    base = _node("draft", score=0.040)
    child = _node("improve", parent=base, score=0.041, plan="DIRECTION: features\nHYPOTHESIS: ranks per era help")
    ledger.register(child, "features")
    ledger.conclude(child, threshold=1e-4)
    text = ledger.prompt_text("model")
    assert "DIRECTION: model" in text and "HYPOTHESIS:" in text
    assert "- features: 1 / 1" in text and "[confirmed] features: ranks per era help" in text
    saved = json.loads((tmp_path / "research_ledger.json").read_text())
    assert saved[0]["verdict"] == "confirmed"


def test_ledger_is_off_unless_enabled():
    agent = SimpleNamespace(scfg=SimpleNamespace(research_loop=False), cfg=SimpleNamespace(log_dir=None))
    assert ledger_for(agent) is None
    agent.scfg.research_loop = True
    ledger = ledger_for(agent)
    assert ledger is not None and ledger_for(agent) is ledger
    assert set(ledger.directions) >= {"targets", "features", "model", "ensemble"}


def test_knowledge_prompt_prefers_checked_entries_for_the_direction(tmp_path):
    from agents.knowledge_context import knowledge_prompt

    entries = [
        {"id": "ks_1", "title": "Tune LightGBM", "principle": "Use 2000 trees.", "category": "tree_models",
         "grade": "A", "n_sources": 9, "stance": "recommend", "check": {"verdict": "helps", "diff": 0.012, "t": 5.6, "eras": 156}},
        {"id": "ks_2", "title": "Neutralize", "principle": "Neutralize to risky features.", "category": "neutralization_and_risk",
         "grade": "A", "n_sources": 50, "stance": "recommend", "check": {"verdict": "hurts", "diff": -0.0035, "t": -3.5, "eras": 156}},
        {"id": "ks_3", "title": "Bigger model", "principle": "Use depth 10.", "category": "tree_models",
         "grade": "B", "n_sources": 2, "stance": "recommend"},
    ]
    path = tmp_path / "ks.json"
    path.write_text(json.dumps({"manifest": {"cutoff": "2024-01-19"}, "entries": entries}))
    agent = SimpleNamespace(cfg=SimpleNamespace(knowledge=SimpleNamespace(
        enabled=True, path=str(path), stages=["draft", "improve"], max_entries=2)))

    text, ids = knowledge_prompt(agent, "improve", "model")
    assert ids == ["ks_1", "ks_3"]
    assert "before 2024-01-19" in text and "helps (CORR +0.0120" in text and "not run" in text

    text, ids = knowledge_prompt(agent, "improve", "post_processing")
    assert ids == ["ks_2"] and "hurts" in text  # only a failed check: shown as a warning

    agent.cfg.knowledge.enabled = False
    assert knowledge_prompt(agent, "draft", None) == ("", [])
