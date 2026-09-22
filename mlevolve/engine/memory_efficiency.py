"""Optional consumption safeguards for lightweight memory, independent of scored outcomes."""
from __future__ import annotations
import ast,json,os,hashlib,time,uuid
from pathlib import Path

def enabled():return os.environ.get('MLEVOLVE_MEMORY_EFFICIENCY')=='1'

def code_fingerprint(code):
    tree=ast.parse(code)
    # Comments and docstrings do not change computation; all executable AST remains.
    class StripDocs(ast.NodeTransformer):
        def strip(self,node):
            self.generic_visit(node)
            if getattr(node,'body',None) and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
            return node
        visit_Module=strip
        visit_FunctionDef=strip
        visit_AsyncFunctionDef=strip
        visit_ClassDef=strip
    return hashlib.sha256(ast.dump(StripDocs().visit(tree),include_attributes=False).encode()).hexdigest()

def duplicate_of(agent,node):
    target=code_fingerprint(node.code)
    for old in getattr(agent.journal,'nodes',[]):
        if old.id==node.id or getattr(old,'stage',None)=='root' or not getattr(old,'code',''):continue
        if code_fingerprint(old.code)==target:return old.id
    return None

def write(agent,name,payload):
    root=Path(agent.cfg.log_dir)/'memory_efficiency';root.mkdir(parents=True,exist_ok=True)
    p=root/(name+'.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+'\n');tmp.replace(p)

def check_plan(agent,request,decision,generate):
    """A second model call checks proposed implementation, not the proposer's self-labels."""
    ids=[a['id'] for a in decision['alternatives']]
    schema={'type':'object','additionalProperties':False,'properties':{
        'action':{'type':'string','enum':['approve','select_other','replan']},
        'selected_id':{'type':'string','enum':ids},'reason':{'type':'string'},
        'evidence':{'type':'array','items':{'type':'string'}},
        'replan_instruction':{'type':'string'}},'required':['action','selected_id','reason','evidence','replan_instruction']}
    payload={'task':request['task'],'source_memories':request['distilled_experience'],
      'current_parent':request['current_parent'],'current_run_evidence':request['current_run_evidence'],
      'alternatives':decision['alternatives'],'proposer_selected_id':decision['selected_id'],
      'review_rules':[
       'Judge the actual proposed changes independently; do not merely echo the proposer memory labels.',
       'Prefer retaining the current-run certified incumbent over drifting along a regressed attempt. Check whether a useful member/stacker is silently removed during a repair.',
       'Check current evidence for repeated unchanged approaches, numerical instability, insufficient convergence, whole-corpus inference, and early checkpoint risks.',
       'You may choose another already-proposed complete plan, or request replanning. Do not invent a new plan, a score, or a universal architecture ban.',
       'At the first draft use applicable prior positive and negative experience, but do not require reproduction of a historical best model.',
       'A genuinely new mitigation or new setting can justify a revisit; below-incumbent is not inherently invalid. Preserve useful stepping stones when justified.',
       'Protocol safety is still independently checked later. Do not require a second full runtime contract or unrelated iMet/session rules.',
       'Return concise JSON only. Cite concrete current/source facts in evidence. Never compare tune loss with a host search score.',
      ]}
    prompt={'system':'Independently check a proposed ML search decision using observed evidence; do not predict unobserved results as facts.','user':json.dumps(payload,ensure_ascii=False,default=str),'assistant':''}
    start=time.monotonic();response=generate(prompt=prompt,temperature=0,cfg=agent.cfg,json_schema=schema);elapsed=time.monotonic()-start
    if isinstance(response,str):
        text=response.strip()
        if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
        response=json.loads(text)
    if not isinstance(response,dict) or response.get('action') not in ('approve','select_other','replan') or response.get('selected_id') not in ids:raise ValueError('invalid independent decision check')
    if not response.get('reason') or not isinstance(response.get('evidence'),list) or not response['evidence']:raise ValueError('decision check requires explicit evidence')
    write(agent,uuid.uuid4().hex+'.plan_check',{'request':payload,'response':response,'seconds':elapsed,'model':agent.acfg.code.model,'no_counterfactual_score_claim':True})
    if response['action']=='replan':raise ValueError('Independent check requests replan: '+response.get('replan_instruction','')+' '+response['reason'])
    result=dict(decision);result['selected_id']=response['selected_id'];result['selection_reason']+='\nIndependent check: '+response['reason'];result['independent_check']=response
    return result

def is_service_failure(error):
    text=str(error).lower()
    return any(x in text for x in ('auth_unavailable','invalidated oauth','quota exceeded','insufficient_quota','usage_limit_reached'))
