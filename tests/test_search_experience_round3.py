from pathlib import Path
import json,importlib.util,types,os,tempfile,copy,unittest,ast
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/spooky_round3_distilled_memory_20260910'
spec=importlib.util.spec_from_file_location('mem3',R/'runtime/mlevolve/engine/search_experience.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PACK=json.loads((R/'memory/search_experience.json').read_text())
class Round3Tests(unittest.TestCase):
 def test_both_sources_have_disjoint_provenance(self):
  self.assertEqual([s['nodes'] for s in PACK['source_runs']],[30,26]);self.assertEqual(len(PACK['transitions']),56)
  self.assertEqual(len({(n['source_run'],n['id']) for n in PACK['transitions']}),56)
  self.assertTrue(all(e['source_round'] in (1,2) for c in PACK['cards'] for e in c['evidence']))
 def test_counterevidence_preserves_parent_improvement(self):
  r=next(n for n in PACK['transitions'] if n['source_round']==2 and n['node']==23)
  self.assertEqual(r['classification'],'improves_parent_but_below_best');self.assertTrue(r['structured_selection_fallback'])
 def test_risk_retrieval_keeps_new_relevant_cards(self):
  ids=[c['id'] for c in m.retrieve(PACK,'deberta label_smoothing rdrop regularization scheduler warmup')]
  self.assertIn('smoothing_after_strong_incumbent_is_unproven',ids)
  self.assertGreaterEqual(sum(c['kind']=='performance_risk' for c in m.retrieve(PACK,'deberta label_smoothing rdrop regularization scheduler warmup logistic window')),3)
 def test_quote_markup_normalizes_but_parameter_values_do_not(self):
  self.assertEqual(m.normalize_quote('“Set `alpha=0.05`. ”'),m.normalize_quote('Set alpha=0.05'))
  self.assertNotEqual(m.normalize_quote('alpha=0.05'),m.normalize_quote('alpha=0.02'))
 def test_normalization_does_not_invent_plans(self):
  d={'alternatives':{x:{'module':'model_design','plan':'change '+x} for x in ('A','B','C')}}
  n=m.normalize_response(d);self.assertEqual(len(n['alternatives']),3);self.assertEqual(n['alternatives'][0]['plan'],{'model_design':'change A'})
  self.assertEqual(m.normalize_response({'alternatives':['bad']}),{'alternatives':['bad']})
 def test_malformed_shapes_fail_with_diagnostic(self):
  for v in ['bad',{'alternatives':['bad']},{'alternatives':{}}]:
   with self.assertRaises(ValueError):m.validate_decision(v,[],['model_design'])
 def test_mitigation_requires_quote_in_actual_plan(self):
  c=next(c for c in PACK['cards'] if c['require_plan_anchor'])
  alts=[{'id':x,'reason':'reason','module':['model_design'],'plan':{'model_design':'Use normalized feature weights '+x},'memory_assessments':[{'memory_id':c['id'],'relation':'mitigated','reason':'controlled balancing','changed_condition':'normalized feature weights','plan_evidence':'invented change outside plan'}],'falsification':'measure degradation','cost_and_reserve':'bounded fit'} for x in ('A','B','C')]
  d={'alternatives':alts,'selected_id':'A','selection_reason':'choose'}
  with self.assertRaisesRegex(ValueError,'plan_evidence'):m.validate_decision(d,[c],['model_design'])
  for a in alts:a['memory_assessments'][0]['plan_evidence']=a['plan']['model_design']
  self.assertEqual(m.validate_decision(d,[c],['model_design'])['id'],'A')
 def test_improve_execution_without_decision_blocked(self):
  with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{'MLEVOLVE_SEARCH_MEMORY_MODE':'on','MLEVOLVE_SEARCH_MEMORY_SHA256':'pack'}):
   a=types.SimpleNamespace(cfg=types.SimpleNamespace(log_dir=t));n=types.SimpleNamespace(id='n',plan='ordinary rewrite',code='print(1)',stage='improve')
   with self.assertRaises(m.MemoryDecisionError):m.record_execution(a,n)
 def test_runtime_catch_raises_before_rewrite(self):
  # Execute the actual guarded except body from the deployed Improve function.
  tree=ast.parse((R/'runtime/mlevolve/agents/improve_agent.py').read_text());handler=next(h for h in ast.walk(tree) if isinstance(h,ast.ExceptHandler) and any(isinstance(n,ast.Raise) and 'MemoryDecisionError' in ast.unparse(n) for n in ast.walk(h)))
  block=ast.fix_missing_locations(ast.Module(body=handler.body,type_ignores=[]))
  import sys
  engine=types.ModuleType('engine');engine.search_experience=m;called=[]
  with patch.dict(sys.modules,{'engine':engine}),patch.dict(os.environ,{'MLEVOLVE_SEARCH_MEMORY_MODE':'on'}):
   with self.assertRaises(m.MemoryDecisionError):exec(compile(block,'actual-except','exec'),{'e':ValueError('bad structure'),'agent':types.SimpleNamespace(cfg=types.SimpleNamespace(log_dir=tempfile.mkdtemp(prefix='round3-guard-test-'))),'parent_node':types.SimpleNamespace(id='parent'),'logger':types.SimpleNamespace(warning=lambda *a:called.append('warning')),'plan_and_code_query':lambda *a:called.append('rewrite')})
  self.assertEqual(called,[])
if __name__=='__main__':unittest.main()
