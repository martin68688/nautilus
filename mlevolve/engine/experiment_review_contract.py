"""Compile review obligations from a frozen experiment and executor contract.

No competition IDs, class counts, thresholds or training schedules are defaults.
The same document is supplied to generation, review and execution admission.
"""
import ast
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_contract(contract):
    if contract.get('schema') != 'experiment-review-contract-v1':
        raise ValueError('unsupported experiment review contract schema')
    for key in ('task', 'data', 'prediction', 'execution', 'resources', 'search'):
        if not isinstance(contract.get(key), dict):
            raise ValueError('missing contract section: ' + key)
    prediction = contract['prediction']
    if prediction.get('kind') not in ('multiclass_probability', 'multilabel', 'regression'):
        raise ValueError('unsupported prediction kind')
    if not prediction.get('columns') or not prediction.get('metric'):
        raise ValueError('prediction columns and metric are required')
    execution = contract['execution']
    if execution.get('interface') not in ('budget_script_v1', 'host_session_v1'):
        raise ValueError('unsupported execution interface')
    if execution['interface'] == 'host_session_v1' and not execution.get('api'):
        raise ValueError('host session executor must supply its actual API contract')
    if prediction['kind'] == 'multiclass_probability':
        if not prediction.get('classes') or len(prediction['columns']) != len(prediction['classes']) + 1:
            raise ValueError('multiclass columns must match the declared class order')
        if contract.get('calibration', {}).get('kind') == 'global_multilabel_threshold':
            raise ValueError('multilabel threshold is incompatible with multiclass probabilities')
    if execution.get('network') not in ('public_pretrained', 'offline'):
        raise ValueError('network policy must be explicit')
    return contract


def load_contract(agent):
    path = str(getattr(agent.cfg, 'semantic_review_contract_path', '') or '')
    expected = str(getattr(agent.cfg, 'semantic_review_contract_sha256', '') or '')
    if not path or not expected:
        raise ValueError('frozen semantic review contract path and SHA256 are required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('semantic review contract SHA256 mismatch')
    contract = validate_contract(json.loads(raw))
    if contract['task']['id'] != str(agent.cfg.exp_id):
        raise ValueError('review contract task does not match this experiment')
    return contract


def compile_rules(contract):
    validate_contract(contract)
    encoded = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    rules = {
        'data_boundaries': 'All learned preprocessing and model updates obey these data roles; no hidden labels or external task solutions: ' + encoded(contract['data']),
        'real_training_prediction': 'Actual model fitting and per-row inference must be reachable. No fabricated scores, dummy predictions or empty successful execution.',
        'metric_and_submission': 'Use this metric, prediction representation, class/column order and ID policy only: ' + encoded(contract['prediction']),
        'plan_implementation': 'Implement the stated candidate plan. Preserve its algorithm choices during protocol repair. Infer no extra model, optimizer, epoch or feature restrictions.',
        'resource_deadlines': 'Respect these limits with bounded training and time reserved for genuine inference and durable saving: ' + encoded(contract['resources']),
        'memory_and_search_scope': 'Follow only this experiment memory/search policy: ' + encoded(contract['search']),
        'external_assets': 'External asset policy: ' + encoded(contract['execution']['network']) + '. Never fetch historical task-specific fitted models, prior run outcomes or solutions.',
    }
    execution = contract['execution']
    if execution['interface'] == 'budget_script_v1':
        rules.update({
            'executable_entrypoint': 'Executor runs python candidate.py. Module execution or a called main function must start the workflow. It never supplies a session or calls solve(session). An uncalled function is invalid. Ordinary imports and startup data reads are allowed within the declared data roles.',
            'checkpoint_lifecycle': 'Instantiate BudgetCheckpoint before expensive work; check should_stop in iterative training or use a bounded fit. Publish after each usable stage. save_checkpoint writes a real self-contained fitted model; predict_checkpoint reloads it and performs genuine inference, without retraining or using a live global fitted model. finish records completion. Library fit stages need not be neural epochs. Closed-form/non-iterative learners may perform one finite fit on the fixed dataset after a stop check, then immediately publish. Do not invent iteration counters for such learners. publish already records its tune score/progress; separate record calls are optional diagnostics.',
        })
    else:
        rules['executable_entrypoint'] = 'Executor invokes solve(session). Use only the actual injected API, scopes and order specified here: ' + encoded(execution['api'])
    if contract.get('training_constraints'):
        rules['training_constraints'] = encoded(contract['training_constraints'])
    if contract.get('calibration'):
        rules['calibration'] = encoded(contract['calibration'])
    return rules


def source_observations(source, contract):
    """Narrow executable-call evidence; semantic judgement remains with reviewer."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [{'code': 'syntax_error', 'line': error.lineno, 'message': error.msg}]
    if contract['execution']['interface'] != 'budget_script_v1':
        return []  # Host-specific adapter owns its static entrypoint checks.
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    aliases = {'BudgetCheckpoint'}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == 'budget_checkpoint':
            aliases.update(a.asname or a.name for a in n.names if a.name == 'BudgetCheckpoint')
    visited, calls = set(), []
    def visit_statements(statements):
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for n in ast.walk(statement):
                if isinstance(n, ast.Call):
                    calls.append(n)
                    if isinstance(n.func, ast.Name) and n.func.id in definitions and n.func.id not in visited:
                        visited.add(n.func.id)
                        visit_statements(definitions[n.func.id].body)
    visit_statements(tree.body)
    targets = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
    methods = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    issues = []
    if not targets.intersection(aliases):
        issues.append({'code': 'unreachable_budget_start', 'line': 0,
                       'message': 'No reachable BudgetCheckpoint construction from script entry. The executor will not call an uncalled solve(session). Trace actual entry before approving.'})
    for method in ('should_stop', 'publish', 'finish'):
        if method not in methods:
            issues.append({'code': 'unreachable_' + method, 'line': 0,
                           'message': 'No reachable ' + method + ' call from script entry; repair the execution path without changing the model.'})
    return issues


def review_allows_execution(report):
    # A disabled reviewer is not a failed review. Actual integrity checks in the
    # executor always remain active, independently of this optional LLM stage.
    return report.get('final_status') == 'clean' or (
        report.get('enabled') is False and report.get('final_status') == 'disabled'
        and report.get('execution_disposition') == 'continue')
