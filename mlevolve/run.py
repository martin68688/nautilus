import atexit
import logging
import os
import signal
import sys
import shutil
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent, SearchSpaceExhausted
from engine.executor import Interpreter
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, save_run_identity, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from engine.run_control import (
    draft_execution_lane,
    focused_outcome_context,
    focused_protocol_status,
    should_continue_focused_search,
)
from engine.run_outcome import (
    FailedRunError,
    PartialRunError,
    classify_run_outcome,
    write_run_outcome,
)
from engine.search_resume import (
    attach_resumed_active_candidates,
    load_search_resume_checkpoint,
    restore_search_workspace,
    write_search_resume_receipt,
)
from utils.logging_config import setup_logging
import torch


class SigtermFinalizer:
    """Seal the latest journal/outcome before Kubernetes terminates the Pod."""

    def __init__(self, cfg, logger, runtime_state):
        self.cfg = cfg
        self.logger = logger
        self.runtime_state = runtime_state

    def persist_interrupted_outcome(self, reason: str) -> None:
        journal = self.runtime_state.get("journal")
        agent = self.runtime_state.get("agent")
        interpreter = self.runtime_state.get("interpreter")
        try:
            if journal is not None:
                save_run(self.cfg, journal)
        except Exception as error:
            self.logger.warning(
                "Termination journal checkpoint failed: %s", error
            )
        active_ids = (
            interpreter.active_candidate_ids()
            if interpreter is not None
            else []
        )
        log_dir = Path(self.cfg.log_dir)
        outcome = classify_run_outcome(
            completed_steps=int(self.runtime_state.get("completed") or 0),
            total_steps=int(self.runtime_state.get("total_steps") or 0),
            search_exhausted=False,
            has_certified_solution=bool(
                agent is not None
                and getattr(agent, "best_node", None) is not None
            ),
            termination_reason=reason,
            active_candidate_ids=active_ids,
            journal_checkpoint_ref=str(
                (log_dir / "journal.json").resolve()
            ),
        )
        try:
            write_run_outcome(log_dir, outcome)
        except Exception as error:
            self.logger.warning(
                "Interrupted outcome was not published: %s", error
            )

    def __call__(self, signum, _frame):
        if self.runtime_state["terminating"]:
            raise SystemExit(128 + int(signum))
        self.runtime_state["terminating"] = True
        self.logger.warning("SIGTERM received; sealing partial run evidence")
        interpreter = self.runtime_state.get("interpreter")
        if interpreter is not None:
            interpreter.terminate_all_subprocesses()
        self.persist_interrupted_outcome("sigterm")
        raise SystemExit(128 + int(signum))



def _run_impl():
    run_started_monotonic = time.monotonic()
    cfg = load_cfg()
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    rng_identity = set_global_seed(cfg.agent.seed)
    cfg.run_identity.rng_state_hash = str(rng_identity["rng_state_hash"])
    cfg.run_identity.rng_state_components = dict(rng_identity)
    logger = setup_logging(cfg)
    resume_checkpoint = load_search_resume_checkpoint(
        total_steps=int(cfg.agent.steps),
    )
    runtime_state = {
        "completed": (
            resume_checkpoint.completed_steps
            if resume_checkpoint is not None
            else 0
        ),
        "total_steps": int(cfg.agent.steps),
        "journal": None,
        "agent": None,
        "interpreter": None,
        "terminating": False,
    }

    signal.signal(signal.SIGTERM, SigtermFinalizer(cfg, logger, runtime_state))
    identity_path = save_run_identity(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')
    logger.info("Run identity persisted before draft generation: %s", identity_path)

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg, task_desc=task_desc)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    if resume_checkpoint is not None:
        restored_dirs = restore_search_workspace(
            resume_checkpoint,
            Path(cfg.workspace_dir),
        )
        receipt_path = write_search_resume_receipt(
            Path(cfg.log_dir),
            resume_checkpoint,
            restored_dirs,
        )
        logger.warning(
            "[resume] loading completed search checkpoint %s/%s from %s; "
            "workspace_dirs=%s receipt=%s",
            resume_checkpoint.completed_steps,
            resume_checkpoint.total_steps,
            resume_checkpoint.source_attempt_root,
            restored_dirs,
            receipt_path,
        )

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = (
        resume_checkpoint.journal
        if resume_checkpoint is not None
        else Journal()
    )
    runtime_state["journal"] = journal
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    if resume_checkpoint is not None:
        # Preserve the original wall-clock search phase.  This keeps time-based
        # routing/fusion decisions and the total frozen budget continuous across
        # immutable attempts instead of granting a fresh full budget.
        agent.search_start_time = (
            time.time() - resume_checkpoint.prior_agent_wall_seconds
        )
        attach_resumed_active_candidates(
            agent, resume_checkpoint.active_candidates
        )
    runtime_state["agent"] = agent

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )
    remaining_agent_wall_seconds = max(
        0.0,
        float(cfg.agent.time_limit)
        - (
            resume_checkpoint.prior_agent_wall_seconds
            if resume_checkpoint is not None
            else 0.0
        ),
    )
    interpreter.set_run_deadline(
        run_started_monotonic + remaining_agent_wall_seconds,
        finalize_reserve_seconds=int(
            getattr(cfg, "finalize_reserve_seconds", 900) or 900
        ),
    )
    runtime_state["interpreter"] = interpreter

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None, focused=False):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(
            exec_callback=exec_callback,
            node=node,
            mandatory_repair_role=dev_execution_role if focused else None,
            excluded_mandatory_repair_role=(
                dev_execution_role if dev_execution_role and not focused else None
            ),
        )

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    strict_draft_roles = bool(getattr(getattr(cfg.agent, "draft_role_policy", None), "enabled", False))
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    lock = threading.Lock()
    completed = len(journal) - 1
    runtime_state["completed"] = completed
    search_exhausted = False
    deadline_reached = False

    dev_execution_role = os.environ.get("RUNFOREST_DEV_EXECUTION_ROLE", "").strip()
    draft_start = (
        int(getattr(agent, "_draft_generation_count", 0))
        if resume_checkpoint is not None
        else 0
    )
    draft_indices = list(range(draft_start, min(initial_draft_count, total_steps)))
    if dev_execution_role:
        configured_roles = [
            agent.configured_draft_role(index)
            for index in range(initial_draft_count)
        ]
        if dev_execution_role not in configured_roles:
            raise RuntimeError(
                "RUNFOREST_DEV_EXECUTION_ROLE must name one configured Draft role; "
                f"role={dev_execution_role!r}, configured={configured_roles}"
            )
        target_index = configured_roles.index(dev_execution_role)
        agent._draft_generation_count = target_index
        draft_indices = [target_index]
        logger.warning(
            "DEV role filter active: generating and executing only role=%s (slot=%s)",
            dev_execution_role,
            target_index,
        )

    generated_draft_nodes = []
    pending_draft_nodes = list(
        resume_checkpoint.active_candidates
        if resume_checkpoint is not None
        else ()
    )
    blocked_draft_nodes = []

    def protocol_focus_status():
        return focused_protocol_status(agent.journal.nodes, dev_execution_role)
    if draft_indices:
        logger.info(
            "📝 Phase 1: Sequential draft generation (code only, slots %s..%s of %s)",
            draft_indices[0] + 1,
            draft_indices[-1] + 1,
            initial_draft_count,
        )

        def step_task_generate_only(draft_idx):
            logger.info(f"[step_task_generate_only] Generating draft from virtual root")
            return agent.step(
                exec_callback=exec_callback,
                node=None,
                execute_immediately=False,
                draft_role=agent.configured_draft_role(draft_idx),
            )

        for draft_idx in draft_indices:
            remaining_work = interpreter.remaining_work_seconds()
            if remaining_work is not None and remaining_work <= 0:
                deadline_reached = True
                logger.warning(
                    "Finalization reserve reached before draft slot %s",
                    draft_idx,
                )
                break
            try:
                logger.info(f"🔨 Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)")
                cur_node = step_task_generate_only(draft_idx)
                generated_draft_nodes.append(cur_node)
                lane = draft_execution_lane(cur_node)
                if lane == "execute":
                    pending_draft_nodes.append(cur_node)
                else:
                    blocked_draft_nodes.append(cur_node)
                    logger.warning(
                        "Draft %s was blocked by pre-execution audit and will enter the repair lane",
                        cur_node.id,
                    )
                save_run(cfg, journal)
                runtime_state["completed"] = len(journal) - 1
                logger.info(f"✅ Draft {draft_idx + 1} code generated: node.id={cur_node.id}, added to virtual_root.children")

            except Exception as e:
                logger.exception(f"❌ Exception during draft {draft_idx + 1} generation: {e}")
                if strict_draft_roles:
                    raise RuntimeError(
                        f"Fixed three-role Draft generation failed at slot {draft_idx}; refusing a partial root set"
                    ) from e

        completed = len(journal) - 1
        runtime_state["completed"] = completed
        logger.info(
            "✅ Phase 1 complete: %s drafts generated (%s executable, %s repair-queued)",
            len(generated_draft_nodes),
            len(pending_draft_nodes),
            len(blocked_draft_nodes),
        )

        if dev_execution_role:
            matching_nodes = [
                node for node in generated_draft_nodes
                if getattr(node, "draft_role", None) == dev_execution_role
            ]
            if len(matching_nodes) != 1:
                raise RuntimeError(
                    "RUNFOREST_DEV_EXECUTION_ROLE must match exactly one generated Draft; "
                    f"role={dev_execution_role!r}, matches={len(matching_nodes)}"
                )
            skipped_roles = [
                getattr(node, "draft_role", None)
                for node in generated_draft_nodes
                if node not in matching_nodes
            ]
            matching_ids = {str(node.id) for node in matching_nodes}
            pending_draft_nodes = [
                node for node in pending_draft_nodes if str(node.id) in matching_ids
            ]
            blocked_draft_nodes = [
                node for node in blocked_draft_nodes if str(node.id) in matching_ids
            ]
            logger.warning(
                "DEV role filter active: executing only role=%s; skipped=%s",
                dev_execution_role,
                skipped_roles,
            )

    if pending_draft_nodes or blocked_draft_nodes or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Pending preflight repairs: {len(blocked_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            agent.begin_search_work()
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None
            finally:
                agent.end_search_work()

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = set()
            focus_futures = set()

            def submit_future(callable_, *args, focused=False):
                future = executor.submit(callable_, *args)
                futures.add(future)
                if focused:
                    focus_futures.add(future)
                return future

            for i, node in enumerate(pending_draft_nodes):
                remaining_work = interpreter.remaining_work_seconds()
                if remaining_work is not None and remaining_work <= 0:
                    deadline_reached = True
                    break
                submit_future(
                    execute_draft_node,
                    node,
                    focused=bool(dev_execution_role and node.draft_role == dev_execution_role),
                )
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            for node in blocked_draft_nodes:
                if not dev_execution_role and completed + len(futures) >= total_steps:
                    break
                is_focused = bool(
                    dev_execution_role and node.draft_role == dev_execution_role
                )
                submit_future(
                    step_task,
                    node,
                    is_focused,
                    focused=is_focused,
                )
                logger.info(
                    "📤 Submitted mandatory preflight repair: %s", node.id
                )

            initial_step_tasks = (
                min(max_workers, total_steps - completed) - len(futures)
            )
            if initial_step_tasks > 0 and not deadline_reached:
                for _ in range(initial_step_tasks):
                    submit_future(step_task)
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            while not deadline_reached and (
                should_continue_focused_search(
                    completed_steps=completed,
                    total_steps=total_steps,
                    status=protocol_focus_status(),
                    focus_in_flight=bool(focus_futures),
                )
                if dev_execution_role
                else completed < total_steps
            ):
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    remaining_work = interpreter.remaining_work_seconds()
                    if (
                        remaining_work is not None
                        and remaining_work <= 0
                        and completed < total_steps
                    ):
                        deadline_reached = True
                        logger.warning(
                            "Finalization reserve reached; terminating active candidates"
                        )
                        interpreter.terminate_all_subprocesses()
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                # Process the focused lane first so its newly queued repair is
                # claimed before an ordinary worker asks for more work.
                for fut in sorted(done, key=lambda item: item not in focus_futures):
                    futures.remove(fut)
                    was_focused = fut in focus_futures
                    focus_futures.discard(fut)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except Exception as e:
                        if isinstance(e, SearchSpaceExhausted):
                            search_exhausted = True
                            logger.warning("Search space exhausted: %s", e)
                        else:
                            logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1  # Exclude virtual node
                        runtime_state["completed"] = completed
                        focus_status = protocol_focus_status()
                        if dev_execution_role and focus_status.completed:
                            logger.warning(
                                "DEV role filter completed its protocol transaction; stopping search"
                            )
                        if completed >= total_steps or focus_status.completed:
                            logger.info(journal_to_string_tree(journal))

                    remaining_work = interpreter.remaining_work_seconds()
                    if (
                        remaining_work is not None
                        and remaining_work <= 0
                        and completed < total_steps
                        and not focus_status.completed
                    ):
                        deadline_reached = True
                    within_shared_budget = (
                        not deadline_reached
                        and completed + len(futures) < total_steps
                    )
                    continue_focused_replay = bool(
                        dev_execution_role
                        and focus_status.active
                        and was_focused
                        and not focus_futures
                    )
                    focus_has_finished = bool(
                        dev_execution_role and focus_status.seen and not focus_status.active
                    )
                    if (
                        not search_exhausted
                        and not focus_has_finished
                        and (within_shared_budget or continue_focused_replay)
                    ):
                        next_is_focused = bool(
                            dev_execution_role
                            and (was_focused or getattr(cur_node, "draft_role", None) == dev_execution_role)
                        )
                        submit_future(
                            step_task,
                            cur_node,
                            next_is_focused,
                            focused=next_is_focused,
                        )
                        logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
                if search_exhausted:
                    for pending in futures:
                        pending.cancel()
                    break
            if deadline_reached:
                for pending in futures:
                    pending.cancel()
                interpreter.terminate_all_subprocesses()
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            interpreter.terminate_all_subprocesses()
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            raise
        except SystemExit:
            interrupted = True
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not interrupted:
                executor.shutdown(wait=True)

    else:
        logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)

    completed = max(completed, len(journal) - 1)
    focused_protocol_repair_expected = bool(
        dev_execution_role
        and any(
            getattr(node, "draft_role", None) == dev_execution_role
            and bool(getattr(node, "audit_repair_required", False))
            for node in generated_draft_nodes
        )
    )
    if dev_execution_role and not focused_protocol_repair_expected:
        logger.warning(
            "DEV focused role completed an execution smoke; no protocol-repair transaction was required"
        )
    focused_scope_complete, focused_termination_reason = focused_outcome_context(
        dev_execution_role,
        protocol_focus_status(),
        require_protocol_repair=focused_protocol_repair_expected,
    )
    run_outcome = classify_run_outcome(
        completed_steps=completed,
        total_steps=total_steps,
        search_exhausted=search_exhausted,
        has_certified_solution=agent.best_node is not None,
        focused_scope_complete=focused_scope_complete,
        termination_reason=(
            "finalization_reserve_reached"
            if deadline_reached
            else focused_termination_reason
        ),
        active_candidate_ids=interpreter.active_candidate_ids(),
        journal_checkpoint_ref=str((cfg.log_dir / "journal.json").resolve()),
    )
    outcome_path = write_run_outcome(cfg.log_dir, run_outcome)
    logger.info(
        "Run outcome persisted: status=%s completed=%s/%s path=%s",
        run_outcome["status"],
        completed,
        total_steps,
        outcome_path,
    )

    try:
        if run_outcome["status"] != "complete":
            logger.warning(
                "Skipping terminal evaluation handoff for non-complete run: %s",
                run_outcome["status"],
            )
            evaluation_request = None
        else:
            from fixed_holdout.handoff import write_evaluation_request

            evaluation_request = write_evaluation_request(
                cfg,
                cfg.log_dir / "journal.json",
                authority=getattr(agent, "evaluation_authority", None),
                selected_node_id=(
                    agent.best_node.id if agent.best_node is not None else None
                ),
                selection_basis=(
                    {
                        "type": "solver_internal_search_metric",
                        "metric_value": agent.best_node.metric.value,
                        "metric_maximize": agent.best_node.metric.maximize,
                        "stage": agent.best_node.stage,
                        "draft_role": agent.best_node.draft_role,
                        "metric_disposition": "search_only",
                        "terminal_metric_observed": False,
                        "formal_rank_claim_authorized": False,
                        "source_score_inherited": False,
                    }
                    if agent.best_node is not None
                    else None
                ),
            )
        if evaluation_request is not None:
            logger.info(
                "Fixed-holdout search finished; external evaluation request: %s",
                evaluation_request,
            )
    except Exception as e:
        logger.error("Failed to write fixed-holdout evaluation request: %s", e)
        raise

    # Adoption tracking: post-run analysis (side-channel, never affects the run itself).
    # Only runs if adoption_tracking.enable + enable_analysis; failure is non-fatal.
    try:
        if getattr(cfg, "adoption_tracking", None) and cfg.adoption_tracking.enable and cfg.adoption_tracking.enable_analysis:
            from analysis.adoption_tracker import run_bounded_adoption_analysis

            adoption_status = run_bounded_adoption_analysis(cfg, journal)
            if adoption_status["status"] != "complete":
                logger.warning(
                    "[adoption_tracker] auxiliary analysis ended with status=%s",
                    adoption_status["status"],
                )
    except Exception as e:
        logger.warning(f"[adoption_tracker] analysis skipped: {e}")

    if run_outcome["status"] == "partial":
        raise PartialRunError(
            f"Run ended partial at {completed}/{total_steps}: {run_outcome['reason']}"
        )
    if run_outcome["status"] == "failed":
        raise FailedRunError(
            f"Run failed at {completed}/{total_steps}: {run_outcome['reason']}"
        )


def run():
    """Run one search and restore the caller's SIGTERM handler on every exit."""

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        return _run_impl()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":    
    run()
