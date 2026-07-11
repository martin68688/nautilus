"""Stage-aware SOP gateway retrieval over the RunForest graph."""

from __future__ import annotations

import collections
import json
import logging
import math
from collections.abc import Callable, Mapping
from typing import Any

from agents.memory.external_skill_memory import RunForestMemoryLayer, _as_list, _tokenize

logger = logging.getLogger("MLEvolve")

PACK_SCHEMA = "stage_hybrid_memory_pack_v1"
RRF_K = 60

STAGE_QUOTAS = {
    "draft": {"sop_candidates": 6, "sop_gateways": 3, "tree_candidates": 2},
    "improve": {"sop_candidates": 4, "sop_gateways": 2, "tree_candidates": 6},
    "debug": {"sop_candidates": 2, "sop_gateways": 1, "tree_candidates": 8},
    "evolution": {"sop_candidates": 6, "sop_gateways": 3, "tree_candidates": 3},
    "fusion": {"sop_candidates": 4, "sop_gateways": 2, "tree_candidates": 4},
}

STAGE_RRF_WEIGHTS = {
    "draft": {"sop": 0.70, "tree": 0.30},
    "improve": {"sop": 0.40, "tree": 0.60},
    "debug": {"sop": 0.25, "tree": 0.75},
    "evolution": {"sop": 0.70, "tree": 0.30},
    "fusion": {"sop": 0.50, "tree": 0.50},
}

STAGE_ROUTE = {
    "draft": "sop_first",
    "improve": "tree_heavy",
    "debug": "tree_first",
    "evolution": "sop_first",
    "fusion": "balanced",
}

STAGE_ALIASES = {
    "multi_fusion": "fusion",
    "fusion_draft": "fusion",
    "aggregation": "fusion",
}


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): item for key, item in value.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Stage-hybrid configuration must be a mapping") from exc


def _merge_quotas(overrides: Any) -> dict[str, dict[str, int]]:
    merged = {stage: values.copy() for stage, values in STAGE_QUOTAS.items()}
    for stage, raw in _plain_mapping(overrides).items():
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in merged:
            raise ValueError(f"Unknown stage quota: {stage}")
        for key, value in _plain_mapping(raw).items():
            if key not in merged[stage] or isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"Invalid stage quota {stage}.{key}={value!r}")
            merged[stage][key] = int(value)
    return merged


def _merge_weights(overrides: Any) -> dict[str, dict[str, float]]:
    merged = {stage: values.copy() for stage, values in STAGE_RRF_WEIGHTS.items()}
    for stage, raw in _plain_mapping(overrides).items():
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in merged:
            raise ValueError(f"Unknown RRF stage: {stage}")
        for key, value in _plain_mapping(raw).items():
            if key not in {"sop", "tree"} or isinstance(value, bool):
                raise ValueError(f"Invalid RRF weight {stage}.{key}={value!r}")
            number = float(value)
            if number < 0:
                raise ValueError(f"Invalid RRF weight {stage}.{key}={value!r}")
            merged[stage][key] = number
        if not math.isclose(sum(merged[stage].values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"RRF weights for {stage} must sum to 1")
    return merged


def weighted_rrf(
    sop_ids: list[str],
    tree_ids: list[str],
    *,
    sop_weight: float,
    tree_weight: float,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse two rankings over execution IDs with deterministic ties."""
    sop_rank = {node_id: rank for rank, node_id in enumerate(sop_ids, 1)}
    tree_rank = {node_id: rank for rank, node_id in enumerate(tree_ids, 1)}
    rows = []
    for node_id in sorted(set(sop_rank) | set(tree_rank)):
        score = 0.0
        if node_id in sop_rank:
            score += sop_weight / (k + sop_rank[node_id])
        if node_id in tree_rank:
            score += tree_weight / (k + tree_rank[node_id])
        rows.append(
            {
                "id": node_id,
                "rrf_score": score,
                "sop_rank": sop_rank.get(node_id),
                "tree_rank": tree_rank.get(node_id),
                "candidate_class": (
                    "sop_transition_matches" if node_id in sop_rank else "tree_only_candidates"
                ),
            }
        )
    return sorted(rows, key=lambda item: (-item["rrf_score"], item["id"]))


class StageAwareHybridMemoryLayer(RunForestMemoryLayer):
    """Opt-in hybrid layer; the existing RunForest layer remains unchanged."""

    def __init__(
        self,
        *args: Any,
        stage_quotas: Any = None,
        rrf_weights: Any = None,
        blocked_run_prefixes: list[str] | None = None,
        gateway_selector: Callable[..., dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = kwargs.get("cfg")
        ext_cfg = getattr(cfg, "external_skill_memory", None) if cfg is not None else None
        if stage_quotas is None and ext_cfg is not None:
            stage_quotas = getattr(ext_cfg, "stage_quotas", None)
        if rrf_weights is None and ext_cfg is not None:
            rrf_weights = getattr(ext_cfg, "rrf_weights", None)
        if blocked_run_prefixes is None and ext_cfg is not None:
            blocked_run_prefixes = getattr(ext_cfg, "blocked_run_prefixes", None)
        self.stage_quotas = _merge_quotas(stage_quotas)
        self.rrf_weights = _merge_weights(rrf_weights)
        self._injected_gateway_selector = gateway_selector
        self._blocked_run_prefixes_override = blocked_run_prefixes
        super().__init__(*args, **kwargs)
        if self.mode != "run_forest_stage_hybrid":
            raise ValueError("StageAwareHybridMemoryLayer requires mode=run_forest_stage_hybrid")
        self._build_sop_reverse_index()

    def _build_sop_reverse_index(self) -> None:
        self._transitions_by_sop: dict[str, list[str]] = collections.defaultdict(list)
        for edge in self.graph.get("edges", []):
            if str(edge.get("kind") or edge.get("type")) != "distills_to":
                continue
            transition_id = str(edge.get("src", ""))
            sop_id = str(edge.get("dst", ""))
            if self.nodes.get(transition_id, {}).get("type") != "Transition":
                continue
            if self.nodes.get(sop_id, {}).get("type") != "SOP":
                continue
            self._transitions_by_sop[sop_id].append(transition_id)
        for values in self._transitions_by_sop.values():
            values.sort()
        meta_prefixes = _as_list((self.graph.get("meta") or {}).get("blocked_run_prefixes"))
        override = self._blocked_run_prefixes_override
        self._blocked_run_prefixes = tuple(str(value) for value in (override if override is not None else meta_prefixes))

    def _positive_transition(self, transition_id: str) -> tuple[bool, str]:
        transition = self.nodes.get(transition_id, {})
        if transition.get("type") != "Transition":
            return False, "not_transition"
        run_id = str(transition.get("run_short_id") or transition.get("run_id") or "")
        if any(run_id.startswith(prefix) for prefix in self._blocked_run_prefixes):
            return False, "blocked_run_prefix"
        if transition.get("quarantined") is True or transition.get("protocol_biased") is True:
            return False, "transition_quarantined_or_protocol_biased"
        child = self.nodes.get(str(transition.get("child_node_id") or ""), {})
        if not self._positive_memory_eligible(child):
            audit = child.get("leakage_audit") if isinstance(child.get("leakage_audit"), dict) else {}
            return False, str(audit.get("memory_disposition") or audit.get("status") or "child_not_code_audited_clean")
        if child.get("quarantined") is True or child.get("protocol_biased") is True:
            return False, "child_quarantined_or_protocol_biased"
        return True, "code_audited_clean"

    def _sop_text_parts(self, node: dict[str, Any]) -> dict[str, str]:
        return {
            "semantic": " ".join(str(node.get(key) or "") for key in ("title", "action", "text")),
            "conditions": " ".join(_as_list(node.get("applies_when")) + _as_list(node.get("condition"))),
            "failures": " ".join(_as_list(node.get("prevents")) + _as_list(node.get("failure_modes"))),
            "evidence": " ".join(_as_list(node.get("evidence_turns")) + _as_list(node.get("source_branches"))),
        }

    def _rank_sops(self, query_text: str, stage: str, limit: int) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query_text)
        rows = []
        for sop_id in self._sops:
            node = self.nodes[sop_id]
            parts = self._sop_text_parts(node)
            scores = {
                key: self._token_overlap(query_tokens, _tokenize(text))
                for key, text in parts.items()
            }
            score = (
                0.50 * scores["semantic"]
                + 0.22 * scores["conditions"]
                + 0.18 * scores["failures"]
                + 0.10 * scores["evidence"]
            )
            if stage == "debug":
                score += 0.12 * scores["failures"]
            clean = []
            rejected = []
            for transition_id in self._transitions_by_sop.get(sop_id, []):
                eligible, reason = self._positive_transition(transition_id)
                (clean if eligible else rejected).append(
                    transition_id if eligible else {"transition_id": transition_id, "reason": reason}
                )
            rows.append(
                {
                    "id": sop_id,
                    "score": score,
                    "score_components": scores,
                    "ranking_backend": "field_aware_lexical",
                    "clean_supporting_transition_ids": clean[:8],
                    "clean_supporting_transition_count": len(clean),
                    "rejected_support": rejected[:8],
                    "rejected_support_count": len(rejected),
                }
            )
        rows.sort(key=lambda item: (-item["score"], item["id"]))
        return rows[:limit]

    def _gateway_function_spec(self):
        from llm import FunctionSpec

        return FunctionSpec(
            name="select_stage_hybrid_sop_gateways",
            description="Select eligible SOP gateway IDs for the current stage.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "gateway_ids": {"type": "array", "items": {"type": "string"}},
                    "reasons": {"type": "object", "additionalProperties": {"type": "string"}},
                    "goal": {"type": "string"},
                },
                "required": ["gateway_ids", "reasons", "goal"],
            },
        )

    def _call_gateway_selector(self, *, stage: str, query_text: str, eligible: list[dict[str, Any]]) -> dict[str, Any]:
        if self._injected_gateway_selector is not None:
            return self._injected_gateway_selector(stage=stage, query_text=query_text, eligible=eligible)
        if self.cfg is None:
            raise RuntimeError("cfg is required for agentic gateway selection")
        from llm import query

        model = getattr(self.cfg.agent.feedback, "model", None) or getattr(self.cfg.agent.code, "model", "")
        return query(
            system_message="Select only supplied clean SOP gateway IDs. Do not invent IDs.",
            user_message=json.dumps({"stage": stage, "query": query_text[-5000:], "eligible": eligible}, ensure_ascii=False),
            model=model,
            temperature=0.0,
            max_tokens=900,
            func_spec=self._gateway_function_spec(),
            cfg=self.cfg,
        )

    def _select_gateways(
        self, candidates: list[dict[str, Any]], *, stage: str, query_text: str, limit: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        eligible = [item for item in candidates if item["clean_supporting_transition_ids"]]
        fallback_ids = [item["id"] for item in eligible[:limit]]
        reasons = {item["id"]: "deterministic eligible score order" for item in eligible[:limit]}
        mode = "deterministic"
        llm_calls = 0
        selected_ids = fallback_ids
        goal = STAGE_ROUTE[stage]
        if self.agentic_enabled and eligible:
            llm_calls = 1
            try:
                result = self._call_gateway_selector(stage=stage, query_text=query_text, eligible=eligible)
                valid_ids = {item["id"] for item in eligible}
                proposed = [str(value) for value in result.get("gateway_ids", [])]
                if not proposed or any(value not in valid_ids for value in proposed):
                    raise ValueError("gateway selector returned missing or ineligible IDs")
                selected_ids = list(dict.fromkeys(proposed))[:limit]
                raw_reasons = result.get("reasons") if isinstance(result.get("reasons"), dict) else {}
                reasons = {node_id: str(raw_reasons.get(node_id, "LLM-selected eligible gateway")) for node_id in selected_ids}
                goal = str(result.get("goal") or goal)
                mode = "llm_validated"
            except Exception as exc:
                mode = "deterministic_fallback"
                logger.warning("[StageHybrid] gateway selector failed; deterministic fallback: %s", exc)
        by_id = {item["id"]: item for item in eligible}
        selected = []
        for node_id in selected_ids:
            row = dict(by_id[node_id])
            row["selection_reason"] = reasons[node_id]
            selected.append(row)
        return selected, {"mode": mode, "llm_tool_calls": llm_calls, "goal": goal, "eligible_count": len(eligible)}

    def _append_unique(self, output: list[str], node_id: str) -> None:
        if node_id in self.nodes and node_id not in output:
            output.append(node_id)

    def _expand_gateways(
        self, selected: list[dict[str, Any]]
    ) -> tuple[list[str], dict[str, list[str]], list[str], list[str], list[dict[str, Any]]]:
        execution_ids: list[str] = []
        gateway_transitions: dict[str, list[str]] = {}
        evidence_refs: list[str] = []
        failure_patterns: list[str] = []
        trace: list[dict[str, Any]] = []
        for gateway in selected:
            sop_id = gateway["id"]
            transitions = list(gateway["clean_supporting_transition_ids"][:2])
            gateway_transitions[sop_id] = transitions
            for transition_id in transitions:
                transition = self.nodes[transition_id]
                expanded_for_transition: list[str] = []
                self._append_unique(execution_ids, transition_id)
                self._append_unique(expanded_for_transition, transition_id)
                parent_id = str(transition.get("parent_node_id") or "")
                child_id = str(transition.get("child_node_id") or "")
                for node_id in (parent_id, child_id):
                    if self._positive_memory_eligible(self.nodes.get(node_id, {})):
                        self._append_unique(execution_ids, node_id)
                        self._append_unique(expanded_for_transition, node_id)
                for node_id in self._ancestor_path(child_id, max_hops=5):
                    if self._positive_memory_eligible(self.nodes.get(node_id, {})):
                        self._append_unique(execution_ids, node_id)
                        self._append_unique(expanded_for_transition, node_id)
                local_best = str(self.nodes.get(child_id, {}).get("local_best_node_id") or "")
                if self._positive_memory_eligible(self.nodes.get(local_best, {})):
                    self._append_unique(execution_ids, local_best)
                    self._append_unique(expanded_for_transition, local_best)
                evidence_refs.extend(self._evidence_by_transition.get(transition_id, []))
                parent = self.nodes.get(child_id, {}).get("parent_id")
                for sibling_id in self._children_by_node.get(str(parent), []):
                    if sibling_id == child_id:
                        continue
                    if not self._positive_memory_eligible(self.nodes[sibling_id]):
                        failure_patterns.extend(self._failure_patterns_by_source.get(sibling_id, []))
                trace.append(
                    {
                        "retrieval_channel": "sop_gateway",
                        "candidate_class": "sop_transition_matches",
                        "gateway_sop_id": sop_id,
                        "supporting_transition_ids": [transition_id],
                        "selection_reason": gateway["selection_reason"],
                        "selection_state": "expanded",
                        "expanded_candidate_ids": expanded_for_transition,
                    }
                )
        return (
            execution_ids,
            gateway_transitions,
            list(dict.fromkeys(evidence_refs))[:12],
            list(dict.fromkeys(failure_patterns))[:12],
            trace,
        )

    def _rank_tree(self, *, stage: str, query_text: str, task_id: str, task_desc: str, limit: int) -> list[str]:
        candidates = [node_id for node_id in self._run_nodes if self._positive_memory_eligible(self.nodes[node_id])]
        stage_bonus = {
            "draft": {"draft": 0.08},
            "improve": {"improve": 0.10, "evolution": 0.05},
            "debug": {"debug": 0.10, "improve": 0.04},
            "evolution": {"evolution": 0.10, "improve": 0.04},
            "fusion": {"improve": 0.05, "evolution": 0.05},
        }[stage]
        return self._rank(
            query_text=query_text,
            candidate_ids=candidates,
            task_id=task_id,
            task_desc=task_desc,
            top_k=limit,
            stage_bonus=stage_bonus,
        )

    def _risk_warnings(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        warnings = []
        for candidate in candidates:
            for rejected in candidate["rejected_support"]:
                warnings.append(
                    {
                        "sop_id": candidate["id"],
                        "transition_id": rejected["transition_id"],
                        "reason": rejected["reason"],
                        "disposition": "warning_or_repair_evidence_only",
                    }
                )
        return warnings[:20]

    def _hybrid_pack(self, *, stage: str, task_id: str, task_desc: str, query_text: str) -> dict[str, Any]:
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_QUOTAS:
            raise ValueError(f"Unsupported stage-hybrid stage: {stage}")
        quotas = self.stage_quotas[stage]
        sop_candidates = self._rank_sops(query_text, stage, quotas["sop_candidates"])
        selected, selection_meta = self._select_gateways(
            sop_candidates, stage=stage, query_text=query_text, limit=quotas["sop_gateways"]
        )
        sop_execution, gateway_transitions, evidence_refs, failure_patterns, trace = self._expand_gateways(selected)
        tree_ids = self._rank_tree(
            stage=stage,
            query_text=query_text,
            task_id=task_id,
            task_desc=task_desc,
            limit=quotas["tree_candidates"],
        )
        weights = self.rrf_weights[stage]
        fused = weighted_rrf(
            sop_execution,
            tree_ids,
            sop_weight=weights["sop"],
            tree_weight=weights["tree"],
        )
        selected_sop_ids = {item["id"] for item in selected}
        for item in sop_candidates:
            trace.append(
                {
                    "retrieval_channel": "sop_direct",
                    "candidate_class": (
                        "sop_transition_matches" if item["id"] in selected_sop_ids else "sop_only_candidates"
                    ),
                    "gateway_sop_id": item["id"] if item["id"] in selected_sop_ids else None,
                    "supporting_transition_ids": item["clean_supporting_transition_ids"],
                    "selection_reason": next(
                        (value["selection_reason"] for value in selected if value["id"] == item["id"]),
                        "not selected as a formal gateway",
                    ),
                    "selection_state": "selected" if item["id"] in selected_sop_ids else "candidate",
                }
            )
        for node_id in tree_ids:
            trace.append(
                {
                    "retrieval_channel": "tree_direct",
                    "candidate_class": (
                        "sop_transition_matches" if node_id in sop_execution else "tree_only_candidates"
                    ),
                    "gateway_sop_id": None,
                    "supporting_transition_ids": [],
                    "selection_reason": f"independent {STAGE_ROUTE[stage]} tree ranking",
                    "selection_state": "selected",
                }
            )
        for item in fused[: self.top_k]:
            trace.append(
                {
                    "retrieval_channel": "hybrid_rrf",
                    "candidate_class": item["candidate_class"],
                    "gateway_sop_id": None,
                    "supporting_transition_ids": [],
                    "selection_reason": f"weighted RRF score={item['rrf_score']:.8f}",
                    "selection_state": "injected",
                    "candidate_id": item["id"],
                }
            )
        sop_only = [item for item in sop_candidates if item["id"] not in selected_sop_ids]
        return {
            "schema": PACK_SCHEMA,
            "stage_route": {"stage": stage, "route": STAGE_ROUTE[stage], "quotas": quotas, "rrf": weights},
            "direct_sop_candidates": sop_candidates,
            "selected_sop_gateways": selected,
            "gateway_transitions": gateway_transitions,
            "tree_candidates": tree_ids,
            "sop_transition_matches": [item for item in fused if item["id"] in sop_execution],
            "sop_only_candidates": sop_only,
            "tree_only_candidates": [item for item in fused if item["id"] not in sop_execution],
            "evidence_refs": evidence_refs,
            "failure_patterns": failure_patterns,
            "risk_warnings": self._risk_warnings(sop_candidates),
            "navigation_trace": trace,
            "fused_execution_candidates": fused,
            "gateway_selection": selection_meta,
        }

    def _format_hybrid_pack(self, pack: dict[str, Any]) -> str:
        compact = {
            key: pack[key]
            for key in (
                "stage_route",
                "selected_sop_gateways",
                "gateway_transitions",
                "sop_transition_matches",
                "sop_only_candidates",
                "tree_only_candidates",
                "evidence_refs",
                "failure_patterns",
                "risk_warnings",
            )
        }
        return "\n".join(
            [
                "## Stage-Aware Hybrid Run-Forest Memory",
                "Candidates are suggestions; verified evidence and risk warnings are separate. Do not treat SOP-only references as proven successful recipes.",
                json.dumps(compact, ensure_ascii=False, indent=2),
            ]
        )

    def retrieve_for_node(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_parts: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        if not self.stage_enabled(stage):
            return "", []
        query_text = "\n".join([task_desc or "", *(query_parts or [])])
        pack = self._hybrid_pack(stage=stage, task_id=task_id, task_desc=task_desc, query_text=query_text)
        self._last_agentic_pack = pack
        refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
        refs += [item["id"] for item in pack["selected_sop_gateways"]]
        refs += pack["evidence_refs"] + pack["failure_patterns"]
        text = self._format_hybrid_pack(pack)
        if self.max_chars > 0 and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n... (stage-hybrid memory truncated)"
        return text, list(dict.fromkeys(refs))
