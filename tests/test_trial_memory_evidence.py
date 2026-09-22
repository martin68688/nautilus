"""Trial export and actual distiller input tests; no teacher/network calls."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "paper-skills/distillation" / (name + ".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
exporter = load("trial_memory_evidence")
distiller = load("distill_recipe_sops")
FIXTURE = ROOT / "experiments/spooky_incumbent_trial_terra_v4_20260910/evidence/trial-memory-real.json"
TASK = "spooky-author-identification"

class TrialMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = json.loads(FIXTURE.read_text())

    def write_package(self):
        p = self.root / "trials.json"
        self.package["payload_sha256"] = exporter.digest({k:v for k,v in self.package.items() if k != "payload_sha256"})
        p.write_text(json.dumps(self.package)); return p

    def test_review_metadata_and_mixed_repairs_keep_correct_attribution(self):
        # Synthetic ledger fixture, not experimental scores or training evidence.
        import hashlib
        run=self.root/'synthetic-run'; logs=run/'logs';logs.mkdir(parents=True)
        config={'architecture':{'description':'old'},'training':{'alpha':.5}}
        updated=copy.deepcopy(config);updated['architecture']['description']='corrected'
        hashes=[hashlib.sha256(s.encode()).hexdigest() for s in ['parent','child']]
        changes=[{'path':'/architecture/description','operation':'replace','before':'old','after':'corrected'}]
        rows=[]
        for i,cfg in enumerate([config,updated]):
            rows.append({'node_id':['p','c'][i],'iteration':i+1,'proposal_base_id':None if i==0 else 'p',
                'configuration':cfg,'configuration_sha256':exporter.digest(cfg),'code_sha256':hashes[i],
                'configuration_changes':[] if i==0 else changes,'is_valid':True,'is_buggy':False,
                'metric':.5 if i==0 else .4,'decision':'keep','stage':'improve','best_so_far':.4})
        rows[1]['intervention']={'kind':'metadata_correction','source_sha256':hashes[1],
            'parent_source_sha256':hashes[0],'changes':changes,'operational_changes':[],
            'declaration_corrections':changes,'implementation_repair':False}
        (logs/'journal.json').write_text(json.dumps({'nodes':[{'id':'p','code':'parent'},{'id':'c','code':'child'}]}))
        (logs/'run_identity.json').write_text('{}')
        def result():
            (logs/'incumbent_search_state.json').write_text(json.dumps({'iterations':rows}))
            return exporter.export(run)['records'][1]
        record=result()
        self.assertEqual(record['kind'],'metadata_correction')
        self.assertEqual(record['attribution'],'no_parameter_effect_claim')
        self.assertFalse(record['verified_score'])
        op={'path':'/training/alpha','operation':'replace','before':.5,'after':.25}
        rows[1]['configuration']['training']['alpha']=.25
        rows[1]['configuration_sha256']=exporter.digest(rows[1]['configuration'])
        rows[1]['configuration_changes']=changes+[op]
        rows[1]['intervention'].update(kind='single_parameter_with_repair',changes=changes+[op],
            operational_changes=[op],implementation_repair=True)
        self.assertEqual(result()['attribution'],'joint_change_observation')
        rows[1]['configuration_changes'][1]['after']=999
        with self.assertRaisesRegex(AssertionError,'diff does not match'):result()

    def test_inline_review_repairs_reach_l3_without_fake_training_nodes(self):
        import hashlib
        record=self.package['records'][0]
        source=next(n['code'] for n in self.package['graph']['nodes'] if n['id']==record['node_id'])
        before=source+'\n# synthetic pre-review fixture\n'
        record['review_repairs']=[{'approved':True,'preserves_chosen_recipe':True,
            'introduces_search':False,'changes_numeric_behavior':False,'kind':'declaration_alignment',
            'reason':'Align the declared classifier probability model with the actual normalized OvR implementation.',
            'before_source':before,'after_source':source,
            'before_source_sha256':hashlib.sha256(before.encode()).hexdigest(),
            'after_source_sha256':hashlib.sha256(source.encode()).hexdigest()}]
        _,_,_,repairs=distiller.load_trial_evidence([self.write_package()])
        self.assertEqual(len(repairs),1)
        self.assertEqual(repairs[0]['failure_node_id'],repairs[0]['successful_node_id'])
        self.assertEqual(repairs[0]['stage_pair'],'pre_execution_review->certified_execution')
        self.assertFalse(repairs[0]['paper_grade_eligible'])
        self.assertIn('Intermediate revisions were not trained',repairs[0]['successful_execution_summary'])
        record['verified_score']=False
        self.assertEqual(distiller.load_trial_evidence([self.write_package()])[3],[])

    def test_real_three_trials_keep_exact_configs_scores_and_epoch_receipts(self):
        observations, admitted, _, repairs = distiller.load_trial_evidence([FIXTURE])
        self.assertEqual((len(observations), len(admitted), len(repairs)), (3,3,0))
        self.assertEqual([r["configuration_after"]["training"]["classifier_alpha"] for r in observations], [1.0,0.5,0.25])
        self.assertTrue(all(r["checkpoint_files_verified"] and r["training_events"] for r in observations))
        self.assertTrue(all(not r["official_score"] for r in observations))
        self.assertTrue(all(not row["paper_grade_eligible"] for row in admitted))

    def test_failed_trial_retained_but_not_positive_parameter_evidence(self):
        failed = self.package["records"][-1]
        failed.update(is_valid=False, is_buggy=True, verified_score=False, metric_after=None,
                      metric_delta=None, comparison="unscored", decision="failed", exception_type="NoSavedCheckpoint")
        observations, admitted, _, _ = distiller.load_trial_evidence([self.write_package()])
        self.assertEqual((len(observations),len(admitted)),(3,2))
        prompt = distiller.build_user_prompt(TASK,admitted,observations)
        self.assertIn("NoSavedCheckpoint",prompt)
        self.assertIn("never proof that its parameter is bad",prompt)

    def test_tampering_and_duplicate_run_snapshots_rejected(self):
        self.package["records"][0]["metric_after"] = 0.001
        with self.assertRaises(AssertionError): exporter.validate_package(self.package)
        with self.assertRaises(ValueError): distiller.load_trial_evidence([FIXTURE,FIXTURE])

    def test_teacher_request_contains_all_trial_values(self):
        observations, admitted, _, _ = distiller.load_trial_evidence([FIXTURE])
        with patch.object(distiller,"_call_teacher_json",return_value=({},{})) as call:
            distiller.call_teacher(base_url="unused",api_key="unused",model="test",task=TASK,
                                   evidence=admitted[:1],trial_observations=observations)
        request = call.call_args.kwargs["user_message"]
        for record in observations:
            self.assertIn(record["trial_id"],request)
            self.assertIn(str(record["metric_after"]),request)

    def test_prepare_stage_preserves_every_trial_despite_recipe_selection_limit(self):
        graph = self.root / "empty.json"; graph.write_text('{"nodes":[]}')
        args = SimpleNamespace(fourtask_graph=graph,spooky_graph=graph,incremental_evidence=None,
                               trial_evidence=[FIXTURE],created_at="test",max_evidence_per_task=1,
                               max_repair_evidence_per_task=1,prepare_only=True)
        with patch.dict(distiller.TASK_DIRECTIONS,{TASK:"minimize"},clear=True):
            evidence,_,_ = distiller.build_artifacts(args)
        self.assertEqual(evidence["trial_observation_count"],3)
        self.assertEqual(evidence["host_verified_trial_admitted_count"],3)
        self.assertEqual(len(evidence["selected_evidence"][TASK]),1)

    def test_budget_failure_never_becomes_reusable_leakage_repair(self):
        records = self.package["records"]
        parent,child = records[0],records[1]
        nodes = {n["id"]:copy.deepcopy(n) for n in self.package["graph"]["nodes"]}
        nodes[parent["node_id"]].update(is_buggy=True,analysis="RunDeadlineExceeded: node budget exhausted")
        transition = {"type":"Transition","outcome":"debug_fixed","parent_buggy":True,
                      "child_buggy":False,"stage_pair":"draft->debug","task":TASK,
                      "parent_node_id":parent["node_id"],"child_node_id":child["node_id"],"text":"Fix the runtime"}
        ok,reason = distiller.repair_transition_eligibility(transition,nodes_by_id=nodes,allowed_tasks={TASK},
                            host_verified_trial_node_ids=frozenset([child["node_id"]]))
        self.assertFalse(ok);self.assertEqual(reason,"infrastructure_failure")

    def test_current_protocol_real_repair_can_use_l3_without_fabricated_paper_grade(self):
        records = self.package["records"]; parent,child = records[0],records[1]
        nodes = {n["id"]:copy.deepcopy(n) for n in self.package["graph"]["nodes"]}
        nodes[parent["node_id"]].update(is_buggy=True,analysis="RuntimeError: tensor device mismatch")
        transition = {"type":"Transition","outcome":"debug_fixed","parent_buggy":True,
                      "child_buggy":False,"stage_pair":"draft->debug","task":TASK,
                      "parent_node_id":parent["node_id"],"child_node_id":child["node_id"],
                      "text":"Move the tensor index onto the target tensor device before indexing"}
        ok,_ = distiller.repair_transition_eligibility(transition,nodes_by_id=nodes,allowed_tasks={TASK},
                            host_verified_trial_node_ids=frozenset([child["node_id"]]))
        self.assertTrue(ok)
        self.assertNotIn("paper_grade_eligible",nodes[child["node_id"]]["leakage_audit"])

if __name__ == "__main__": unittest.main(verbosity=2)
