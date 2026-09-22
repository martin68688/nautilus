"""Cross-task contract compilation and execution-path regression checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'mlevolve/engine/experiment_review_contract.py'
if not MODULE.exists():
    MODULE = Path(__file__).resolve().parents[1] / 'mlevolve/engine/experiment_review_contract.py'
spec = importlib.util.spec_from_file_location('review_contract', MODULE)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

BASE = {
    'schema': 'experiment-review-contract-v1',
    'task': {'id': 'text-example'},
    'data': {'fit': 'fit only', 'search': 'host labels only'},
    'prediction': {'kind': 'multiclass_probability', 'classes': ['a', 'b'],
                   'columns': ['id', 'a', 'b'], 'metric': 'log_loss'},
    'execution': {'interface': 'budget_script_v1', 'network': 'public_pretrained'},
    'resources': {'node_total_seconds': 3600},
    'search': {'memory_enabled': False, 'architecture': 'free'},
}
GOOD = '''from budget_checkpoint import BudgetCheckpoint
def main():
    budget = BudgetCheckpoint()
    if not budget.should_stop():
        budget.publish(tune_probs=p, save_checkpoint=save, predict_checkpoint=predict, step=1)
    budget.finish()
if __name__ == '__main__': main()
'''


class ReviewContracts(unittest.TestCase):
    def test_multiclass_has_no_inherited_multilabel_rules(self):
        rules = review.compile_rules(BASE)
        self.assertNotIn('calibration', rules)
        self.assertNotIn('training_constraints', rules)
        text = json.dumps(rules)
        for stale in ('attribute_ids', '3473', 'MIN_EPOCHS', '0.02..0.50', 'image corpus'):
            self.assertNotIn(stale, text)

    def test_other_task_gets_only_its_explicit_constraints(self):
        c = copy.deepcopy(BASE)
        c['task']['id'] = 'unseen-image-task'
        c['prediction'] = {'kind': 'multilabel', 'columns': ['key', 'labels'], 'metric': 'micro_f1'}
        c['execution'] = {'interface': 'host_session_v1', 'network': 'offline', 'api': {'entrypoint': 'solve(session)'}}
        c['calibration'] = {'kind': 'global_multilabel_threshold', 'grid': [0.1, 0.3]}
        c['training_constraints'] = {'max_epochs': 7}
        rules = review.compile_rules(c)
        self.assertIn('0.3', rules['calibration'])
        self.assertIn('7', rules['training_constraints'])
        self.assertNotIn('checkpoint_lifecycle', rules)
        self.assertNotIn('3473', json.dumps(rules))
        c['prediction'] = {'kind': 'regression', 'columns': ['id', 'value'], 'metric': 'rmse'}
        del c['calibration']
        self.assertIn('rmse', review.compile_rules(c)['metric_and_submission'])

    def test_incompatible_contract_rejected_before_review(self):
        c = copy.deepcopy(BASE)
        c['calibration'] = {'kind': 'global_multilabel_threshold'}
        with self.assertRaises(ValueError): review.validate_contract(c)
        c = copy.deepcopy(BASE); c['execution']['interface'] = 'unknown'
        with self.assertRaises(ValueError): review.validate_contract(c)

    def test_uncalled_solve_is_not_an_executable_candidate(self):
        self.assertEqual(review.source_observations(GOOD, BASE), [])
        source = GOOD.replace('def main():', 'def solve(session):').split("if __name__")[0]
        codes = {x['code'] for x in review.source_observations(source, BASE)}
        self.assertIn('unreachable_budget_start', codes)
        self.assertIn('unreachable_publish', codes)

    def test_free_methods_not_required_to_have_epochs(self):
        source = GOOD.replace('main()', 'train_arbitrary_pipeline()')
        self.assertEqual(review.source_observations(source, BASE), [])

    def test_disabled_is_distinct_from_rejected_or_unavailable(self):
        self.assertTrue(review.review_allows_execution({'enabled': False, 'final_status': 'disabled', 'execution_disposition': 'continue'}))
        for status in ('revise', 'uncertain', 'unavailable', 'syntax_error', 'repair_invalid'):
            self.assertFalse(review.review_allows_execution({'final_status': status}))
        self.assertFalse(review.review_allows_execution({'enabled': True, 'final_status': 'disabled'}))

    def test_contract_hash_and_task_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'contract.json'; raw = json.dumps(BASE).encode(); path.write_bytes(raw)
            cfg = SimpleNamespace(exp_id='text-example', semantic_review_contract_path=str(path),
                                  semantic_review_contract_sha256=hashlib.sha256(raw).hexdigest())
            agent = SimpleNamespace(cfg=cfg)
            self.assertEqual(review.load_contract(agent), BASE)
            cfg.exp_id = 'wrong-task'
            with self.assertRaises(ValueError): review.load_contract(agent)
            cfg.exp_id = 'text-example'; path.write_text('{}')
            with self.assertRaises(ValueError): review.load_contract(agent)


if __name__ == '__main__': unittest.main()
