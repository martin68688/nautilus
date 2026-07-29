"""Build guidance description for agent from task/model JSON."""
import json
import re
from pathlib import Path
from typing import Dict, List, Any

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"
METHODOLOGY_MAP_JSON = Path(__file__).resolve().parent / "methodology_map.json"

# Side-channel: most recent methodology ref_ids (set by build_guidance_description,
# read by AgentSearch.__init__ for adoption tracking). NEVER injected into prompts.
_LAST_REF_IDS: list[str] = []
_LAST_RUN_FOREST_REF_IDS: list[str] = []
_LAST_RUN_FOREST_SOURCE: str = ""
_LAST_RUN_FOREST_TEXT: str = ""
_LAST_PRIMARY_MODEL_NAME: str = ""
_LAST_PRIMARY_MODEL_TEXT: str = ""
_LAST_METHODOLOGY_CANDIDATES: list[dict[str, str]] = []


def _looks_like_run_forest_memory(ext_cfg: Any) -> bool:
    if ext_cfg is None or not getattr(ext_cfg, "enable", False):
        return False
    values = [
        getattr(ext_cfg, "mode", ""),
        getattr(ext_cfg, "source_name", ""),
        getattr(ext_cfg, "graph_path", ""),
    ]
    return any("run_forest" in str(value).lower() for value in values)


def _build_run_forest_coldstart_text(cfg: Any, task_desc: str) -> tuple[str, list[str], str]:
    """Return a read-only Run-Forest map path pack for initial draft guidance."""
    ext_cfg = getattr(cfg, "external_skill_memory", None)
    if not _looks_like_run_forest_memory(ext_cfg):
        return "", [], ""
    source_name = getattr(
        ext_cfg,
        "source_name",
        "run_forest_agentic_memory",
    )
    # Bundle-backed memory is not legal to inspect before AgentSearch creates
    # its Authority adapter and hash-verified MemorySnapshot.  Runtime draft
    # retrieval happens after that boundary, so defer instead of touching the
    # legacy graph_path and creating an unledgered pre-prompt exposure.
    if str(getattr(ext_cfg, "bundle_root", "") or "").strip():
        return "", [], str(source_name)
    try:
        from agents.memory.external_skill_memory import RunForestMemoryLayer

        layer = RunForestMemoryLayer(
            graph_path=getattr(ext_cfg, "graph_path", ""),
            index_path=getattr(ext_cfg, "index_path", ""),
            source_name=getattr(ext_cfg, "source_name", "run_forest_agentic_memory"),
            mode=getattr(ext_cfg, "mode", "run_forest_agentic"),
            scoring_mode=getattr(ext_cfg, "scoring_mode", "poincare"),
            enable_agentic=getattr(ext_cfg, "enable_agentic", False),
            navigator_max_steps=getattr(ext_cfg, "navigator_max_steps", 3),
            navigator_reference_budget=getattr(ext_cfg, "navigator_reference_budget", 1200),
            top_k=min(int(getattr(ext_cfg, "top_k", 6) or 6), 6),
            max_chars=min(int(getattr(ext_cfg, "max_chars", 5000) or 5000), 4500),
            cfg=cfg,
        )
        text, ref_ids = layer.retrieve_for_node(
            stage="draft",
            task_id=getattr(cfg, "exp_id", ""),
            task_desc=task_desc or getattr(cfg, "exp_id", ""),
            query_parts=["cold-start task-level successful branches"],
        )
        if not text:
            return "", [], layer.source_name
        guidance = (
            "\n\n---\n## Run-Forest Cold-Start Map Path Pack\n"
            "Before the first draft, a read-only Memory Navigator inspected historical run trees. "
            "Use these paths as evidence-backed starting hints, not as commands to copy blindly.\n\n"
            f"{text}"
        )
        return guidance, ref_ids, layer.source_name
    except Exception as exc:
        return (
            "\n\n---\n## Run-Forest Cold-Start Map Path Pack\n"
            f"Run-Forest cold-start memory was configured but unavailable; continuing without it. Reason: {exc}",
            [],
            source_name if ext_cfg is not None else "run_forest_agentic_memory",
        )


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_models_for_task(
    task_name: str, tasks: Dict, models: Dict
) -> List[Dict[str, str]]:
    """Match model list for task from knowledge by task name."""
    if task_name not in tasks:
        return []
    category = tasks[task_name]  # flat string: "General Image", "NLP", etc.
    if category not in models:
        return []
    matched = []
    for m_name, m_info in models[category].items():
        matched.append({
            "model_name": m_name,
            "description": m_info.get("Description", ""),
            "code_template": m_info.get("Code_template", ""),
        })
    return matched


def _format_model_guidance(model: Dict[str, str], index: int) -> str:
    return "".join([
        f"\nModel{index}: {model['model_name']}\n",
        f"Description:{model['description']}\n",
        "Code template (MUST copy exactly — do NOT change model variant names or file paths):\n```python\n",
        model["code_template"],
        "\n```",
    ])


def _build_guidance_text(task_name: str, tasks: Dict, models: Dict) -> str:
    """Build guidance text from task name and knowledge."""
    model_list = collect_models_for_task(task_name, tasks, models)
    if not model_list:
        return "None model"
    lines = []
    for i, m in enumerate(model_list):
        lines.append(_format_model_guidance(m, i + 1))
    return "\n".join(lines)


def get_init_solution_paths(exp_id: str) -> List[str]:
    """Load init solution paths for exp_id from engine/coldstart/init_solution_paths.json."""
    if not INIT_SOLUTION_JSON.exists():
        return []
    try:
        data = _load_json(str(INIT_SOLUTION_JSON))
        paths = data.get(exp_id)
        if isinstance(paths, list):
            return [str(p) for p in paths if p]
        return []
    except Exception:
        return []


def _extract_positive_sections(text: str) -> list[str]:
    """Extract ## [POSITIVE] sections from a methodology md file."""
    sections = []
    pattern = re.compile(r'^## \[POSITIVE\] (.+?)$\n(.*?)(?=^## \[|^# [^#]|\Z)', re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        title = match.group(1).strip()
        body = match.group(2).strip()
        sections.append(f"**[POSITIVE] {title}**\n{body}")
    return sections


def _build_methodology_text(task_name: str, methodology_kb_path: str) -> tuple[str, list[str]]:
    """Extract only [POSITIVE] entries from original methodology files.

    Returns (text, ref_ids). ref_ids are side-channel ids for adoption tracking.
    """
    if not METHODOLOGY_MAP_JSON.exists():
        return "", []
    try:
        mapping = _load_json(str(METHODOLOGY_MAP_JSON))
    except Exception:
        return "", []

    folders = mapping.get(task_name, [])
    if not folders:
        return "", []

    kb_base = Path(methodology_kb_path)
    all_entries = []
    ref_ids = []  # side-channel
    for folder in folders:
        cat_dir = kb_base / folder
        if not cat_dir.exists():
            continue
        for md_file in sorted(cat_dir.glob("*_methodology.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            entries = _extract_positive_sections(text)
            all_entries.extend(entries)
            ref_ids.append(f"static:{folder}/{md_file.stem}")

    if not all_entries:
        return "", []

    text = (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following actionable techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(all_entries)
    )
    return text, ref_ids


def _build_static_methodology_candidates(
    task_name: str, methodology_kb_path: str
) -> list[dict[str, str]]:
    """Decompose legacy static methodology into Claim-use candidates."""

    import hashlib

    if not METHODOLOGY_MAP_JSON.exists():
        return []
    try:
        mapping = _load_json(str(METHODOLOGY_MAP_JSON))
    except Exception:
        return []
    output: list[dict[str, str]] = []
    kb_base = Path(methodology_kb_path)
    for folder in mapping.get(task_name, []):
        for md_file in sorted((kb_base / folder).glob("*_methodology.md")):
            try:
                sections = _extract_positive_sections(
                    md_file.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            for index, section in enumerate(sections):
                ref_id = f"static:{folder}/{md_file.stem}#{index}"
                digest = hashlib.sha256(
                    f"{ref_id}\n{section}".encode("utf-8")
                ).hexdigest()
                output.append(
                    {
                        "candidate_id": f"methodology::{digest[:24]}",
                        "claim_id": f"methodology_claim::{digest[:24]}",
                        "ref_id": ref_id,
                        "category": str(folder),
                        "title": section.splitlines()[0],
                        "text": section,
                        "content_sha256": hashlib.sha256(
                            section.encode("utf-8")
                        ).hexdigest(),
                    }
                )
    return output


def build_guidance_description(cfg: Any, task_desc: str = "") -> str:

    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    primary_models = collect_models_for_task(cfg.exp_id, tasks, models)
    text = _build_guidance_text(cfg.exp_id, tasks, models)
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))

    methodology_kb_path = getattr(cfg, "methodology_kb_path", "") or ""
    ref_ids: list[str] = []
    methodology_candidates: list[dict[str, str]] = []
    if methodology_kb_path:
        use_dynamic = getattr(cfg, "methodology_dynamic", False)
        if use_dynamic and task_desc:
            from engine.coldstart.methodology_agent import build_methodology_candidates

            methodology_candidates = build_methodology_candidates(
                task_desc, methodology_kb_path, cfg.agent.code
            )
        else:
            methodology_candidates = _build_static_methodology_candidates(
                cfg.exp_id, methodology_kb_path
            )
        ref_ids = [item["ref_id"] for item in methodology_candidates]

    # Keep model-template cold start byte-compatible with the original path.
    # Run-Forest cold-start memory is injected later as a separate external
    # memory section so the "copy template exactly" rule still refers only to
    # the original model template text.
    run_forest_text, run_forest_ref_ids, run_forest_source = _build_run_forest_coldstart_text(cfg, task_desc)

    global _LAST_REF_IDS, _LAST_RUN_FOREST_REF_IDS, _LAST_RUN_FOREST_SOURCE, _LAST_RUN_FOREST_TEXT
    global _LAST_PRIMARY_MODEL_NAME, _LAST_PRIMARY_MODEL_TEXT
    global _LAST_METHODOLOGY_CANDIDATES
    _LAST_REF_IDS = ref_ids  # side-channel snapshot for adoption tracking
    _LAST_RUN_FOREST_REF_IDS = list(run_forest_ref_ids)
    _LAST_RUN_FOREST_SOURCE = run_forest_source
    _LAST_RUN_FOREST_TEXT = run_forest_text
    if primary_models:
        _LAST_PRIMARY_MODEL_NAME = primary_models[0]["model_name"]
        _LAST_PRIMARY_MODEL_TEXT = _format_model_guidance(primary_models[0], 1)
        if torch_hub_dir:
            _LAST_PRIMARY_MODEL_TEXT = _LAST_PRIMARY_MODEL_TEXT.replace(
                "{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/")
            )
    else:
        _LAST_PRIMARY_MODEL_NAME = ""
        _LAST_PRIMARY_MODEL_TEXT = "None model"
    # Methodology is deliberately *not* appended here. AgentSearch creates its
    # Authority adapter later and materializes only the admitted Claim-use text.
    _LAST_METHODOLOGY_CANDIDATES = [dict(item) for item in methodology_candidates]
    return text
