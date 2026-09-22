"""Regression checks for source evidence, actual planner injection and controls."""
from pathlib import Path
import copy,importlib.util,json,os,sys,tempfile,types,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/spooky_round2_distilled_memory_20260910'
spec=importlib.util.spec_from_file_location('search_experience',ROOT/'mlevolve/engine/search_experience.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PACK=json.loads((PACKAGE/'memory/search_experience.json').read_text())
MODULES=['model_design','training_evaluation']

def response(cards,selected='B'):
    items=[]
    for k in ['A','B','C']:
        items.append({'id':k,'reason':'A distinct measured-cost hypothesis '+k,'module':['model_design'],
          'plan':{'model_design':'Change '+k+' with an explicit runtime mitigation'},
          'memory_assessments':[{'memory_id':c['id'],'relation':'avoid_repeat' if k=='A' and c['kind']=='performance_risk' else 'not_applicable',
             'reason':'Same conditions' if k=='A' else 'This hypothesis changes a different factor','changed_condition':''} for c in cards],
          'falsification':'Compare observed tune/search and time after a real training run','cost_and_reserve':'Bound fitting and reserve publication time'})
    return {'alternatives':items,'selected_id':selected,'selection_reason':'Prefer a justified change over an unchanged observed regression'}

class SearchExperienceTests(unittest.TestCase):
 def test_source_outcomes_do_not_conflate_risks(self):
    by={n['node']:n for n in PACK['transitions']}
    self.assertEqual(by[13]['classification'],'improves_parent_but_below_best')
    self.assertEqual(by[16]['classification'],'saved_fallback_not_method_result')
    self.assertTrue(all(by[i]['classification']=='budget_not_executed' for i in range(24,31)))
    self.assertEqual(len(PACK['cards']),12)
 def test_all_failures_keep_rate_bounded(self):
    result=m.summarize([{'is_valid':False,'metric':None} for _ in range(3)])
    self.assertEqual(result['nonrefresh_rate'],1.0)
 def test_task_split_and_digest_bindings(self):
    path=PACKAGE/'memory/search_experience.json';h=m.sha(path.read_bytes())
    self.assertEqual(m.load(path,h,PACK['task_id'],PACK['data_manifest_sha256']),PACK)
    for args in [('bad',PACK['task_id'],PACK['data_manifest_sha256']),(h,'other',PACK['data_manifest_sha256']),(h,PACK['task_id'],'wrong_split')]:
      with self.assertRaises(ValueError):m.load(path,*args)
 def test_context_retrieval_and_no_origin_replay(self):
    cs=m.retrieve(PACK,'deberta dual prefix suffix window',limit=4)
    self.assertIn('avoid_unchanged_dual_window',[c['id'] for c in cs])
    self.assertTrue(all(c['kind'] in ('engineering_risk','search_principle') for c in m.retrieve(PACK,'',stage='draft')))
 def test_known_unchanged_risk_requires_alternative_or_mitigation(self):
    cards=[PACK['cards'][0]];v=response(cards,'A')
    with self.assertRaisesRegex(ValueError,'unchanged'):m.validate_decision(v,cards,MODULES)
    a=v['alternatives'][0]['memory_assessments'][0];a['relation']='mitigated'
    with self.assertRaisesRegex(ValueError,'changed condition'):m.validate_decision(v,cards,MODULES)
    a['changed_condition']='Use measured normalized view weights and independently selected regularization; prior trial lacked them'
    self.assertEqual(m.validate_decision(v,cards,MODULES)['id'],'A')
 def test_unknown_or_unassessed_memories_rejected(self):
    cards=[PACK['cards'][0]];v=response(cards);v['alternatives'][0]['memory_assessments']=[]
    with self.assertRaises(ValueError):m.validate_decision(v,cards,MODULES)
    v=response(cards);v['alternatives'][0]['memory_assessments'][0]['memory_id']='made_up'
    with self.assertRaises(ValueError):m.validate_decision(v,cards,MODULES)
 def setup_planner(self,temp,mode):
    public=Path(temp)/'public';public.mkdir();(public/'manifest.json').write_text('{"test":true}')
    pack=copy.deepcopy(PACK);pack['data_manifest_sha256']=m.sha((public/'manifest.json').read_bytes());p=Path(temp)/'pack.json';p.write_text(json.dumps(pack))
    agent=types.SimpleNamespace(cfg=types.SimpleNamespace(exp_id=PACK['task_id'],log_dir=temp),acfg=types.SimpleNamespace(code=types.SimpleNamespace(model='gpt-5.6-terra',temp=0)))
    env={'MLEVOLVE_SEARCH_MEMORY_MODE':mode,'MLEVOLVE_SEARCH_MEMORY_PATH':str(p),'MLEVOLVE_SEARCH_MEMORY_SHA256':m.sha(p.read_bytes()),'MLEVOLVE_PUBLIC_DATA':str(public)}
    return agent,env
 def test_actual_framework_planner_request_contains_memory_and_links_code(self):
    with tempfile.TemporaryDirectory() as tmp:
      agent,env=self.setup_planner(tmp,'on');requests=[]
      def generate(**kwargs):
        q=json.loads(kwargs['prompt']['user']);requests.append(q)
        return response(q['distilled_experience']['cards'])
      llm=types.ModuleType('llm');llm.generate=generate;llm.compile_prompt_to_md=lambda *args:''
      profiles=types.ModuleType('llm.model_profiles');profiles.thinking_json_incompatible=lambda *a:False
      engine=types.ModuleType('engine');engine.search_experience=m
      with patch.dict(os.environ,env),patch.dict(sys.modules,{'llm':llm,'llm.model_profiles':profiles,'engine':engine}):
        s=importlib.util.spec_from_file_location('tested_planner',PACKAGE/'runtime/mlevolve/agents/planner/base_planner.py');planner=importlib.util.module_from_spec(s);s.loader.exec_module(planner)
        ctx={'stage':'improve','previous_code':'deberta dual prefix suffix window','parent_node':types.SimpleNamespace(id='parent-21'),'execution_output':'current score'}
        result=planner.run_planner(agent,{'Task description':'test task','Instructions':{}},'',ctx,'task','suffix')
        self.assertTrue(requests[0]['distilled_experience']['cards']);self.assertIn('search_memory_decision',result)
        node=types.SimpleNamespace(id='child',code='print(1)',stage='improve',plan=json.dumps(result));m.record_execution(agent,node)
        link=json.loads((Path(tmp)/'search_memory/child.execution_link.json').read_text())
        self.assertEqual(link['decision']['decision_id'],result['search_memory_decision']['decision_id'])
        self.assertEqual(link['code_sha256'],m.sha(node.code))
 def test_control_does_not_load_source_memory(self):
    with tempfile.TemporaryDirectory() as tmp:
      agent,env=self.setup_planner(tmp,'control');env['MLEVOLVE_SEARCH_MEMORY_PATH']='/missing';env['MLEVOLVE_SEARCH_MEMORY_SHA256']='invalid'
      def generate(**kwargs):
        q=json.loads(kwargs['prompt']['user']);self.assertEqual(q['distilled_experience']['cards'],[]);return response([])
      with patch.dict(os.environ,env):r=m.plan(agent,{'Task description':'task','Instructions':{}},{'previous_code':'code'},MODULES,generate)
      self.assertIsNone(r['search_memory_decision']['pack_sha256'])
 def test_unvalidated_response_never_silently_falls_back(self):
    with tempfile.TemporaryDirectory() as tmp:
      agent,env=self.setup_planner(tmp,'on');calls=[]
      def generate(**kwargs):calls.append(1);return {}
      with patch.dict(os.environ,env),self.assertRaises(RuntimeError):m.plan(agent,{'Task description':'task','Instructions':{}},{'previous_code':'deberta'},MODULES,generate)
      self.assertEqual(len(calls),2)
 def test_budget_admission_stop_prevents_repair_generation(self):
    s=importlib.util.spec_from_file_location('tested_linear',PACKAGE/'runtime/mlevolve/engine/linear_search.py');linear=importlib.util.module_from_spec(s);s.loader.exec_module(linear)
    class Exhausted(Exception):pass
    ag=types.ModuleType('engine.agent_search');ag.SearchSpaceExhausted=Exhausted
    en=types.ModuleType('engine');en.solution_manager=types.SimpleNamespace()
    metric=types.ModuleType('utils.metric');metric.WorstMetricValue=object
    with patch.dict(sys.modules,{'engine':en,'engine.agent_search':ag,'utils.metric':metric}),patch.object(linear,'validate',lambda a:None),patch.dict(os.environ,{'MLEVOLVE_BUDGET_ADMISSION_CLOSED':'1'}):
      with self.assertRaisesRegex(Exhausted,'without futile'):linear.step(types.SimpleNamespace(),lambda *a:None)
 def test_metrics_preserve_every_failure_and_regression(self):
    source=json.loads((ROOT/'reports/spooky_terra_round1_results_20260910/source/run_snapshot.json').read_text())
    s=m.summarize(source['state']['iterations']);self.assertEqual(s['attempts'],30);self.assertEqual(s['valid'],17);self.assertEqual(s['failed'],13)
    self.assertEqual(s['new_best_after_initial'],7);self.assertEqual(s['nonrefresh_count'],22);self.assertIsNone(s['avoided_training_failures_proven'])

if __name__=='__main__':unittest.main()
