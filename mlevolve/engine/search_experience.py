"""Opt-in evidence-conditioned candidate planning. Historical losses are never live metrics.

This module owns retrieval, a bounded alternative-selection protocol and audit
receipts. It does not relax semantic review or fabricate avoided-loss estimates.
"""
from __future__ import annotations
import copy
import ast
import re
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

SCHEMA='mlevolve_search_experience_v1'

class MemoryDecisionError(RuntimeError):
    pass


def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def enabled():
    mode=os.environ.get('MLEVOLVE_SEARCH_MEMORY_MODE','off')
    if mode not in ('on','off','control'):raise ValueError('unknown search memory mode')
    return mode=='on'


def planning_enabled():
    enabled()  # Validate the mode even for control.
    return os.environ.get('MLEVOLVE_SEARCH_MEMORY_MODE','off') in ('on','control')


def load(path,expected,task_id,data_sha):
    raw=Path(path).read_bytes()
    if not expected or sha(raw)!=expected:raise ValueError('search memory SHA256 mismatch')
    pack=json.loads(raw)
    if pack.get('schema')!=SCHEMA:raise ValueError('unsupported search memory schema')
    if pack['task_id']!=task_id:raise ValueError('search memory task mismatch')
    if pack['data_manifest_sha256']!=data_sha:raise ValueError('search memory data split mismatch')
    ids=[c['id'] for c in pack['cards']]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('invalid memory IDs')
    for c in pack['cards']:
        for key in ('kind','when','lesson','alternative','limits','evidence','match_terms'):
            if key not in c:raise ValueError('incomplete memory card: '+key)
        if c['kind']=='performance_risk' and not any(e.get('score') is not None for e in c['evidence']):
            raise ValueError('unexecuted proposals cannot establish performance risk')
    return pack


def load_for_agent(agent):
    data=Path(os.environ['MLEVOLVE_PUBLIC_DATA'])/'manifest.json'
    return load(os.environ['MLEVOLVE_SEARCH_MEMORY_PATH'],os.environ['MLEVOLVE_SEARCH_MEMORY_SHA256'],
                str(agent.cfg.exp_id),sha(data.read_bytes()))


def retrieve(pack,text,stage='improve',limit=8):
    text=text.lower();cards=pack['cards'];ranked=[]
    for order,c in enumerate(cards):
        terms=[term.lower() for term in c['match_terms'] if term.lower() in text]
        if stage=='draft' and c['kind'] not in ('engineering_risk','search_principle'):continue
        if not terms and stage!='draft' and c['kind'] not in ('engineering_risk','search_principle'):continue
        score=sum(1+math.log((len(cards)+1)/(1+sum(t in [x.lower() for x in k['match_terms']] for k in cards))) for t in terms)
        score += 8*sum(t.lower() in text for t in c.get('salient_terms',[]))
        ranked.append((score,-order,c))
    ranked=sorted(ranked,key=lambda x:(x[0],x[1]),reverse=True)
    # Keep context-matching risks visible; no forced bad proposal or forced rejection.
    risks=[c for score,_,c in ranked if c['kind']=='performance_risk' and score>0][:3] if stage!='draft' else []
    selected=risks[:limit]
    for _,_,c in ranked:
        if len(selected)>=limit:break
        if c not in selected:selected.append(c)
    return selected


def project(card):
    return {**{k:card[k] for k in ('id','kind','when','lesson','alternative','limits','confidence')},
            'observed_source_evidence':[{k:e.get(k) for k in ('source_round','source_run','node','score','incumbent_before','gap_to_incumbent','classification','structured_selection_fallback')} for e in card['evidence']]}


def memory_block(agent,text='',stage='draft'):
    if not enabled():return {}
    pack=load_for_agent(agent);cards=retrieve(pack,text,stage)
    return {'pack_sha256':os.environ['MLEVOLVE_SEARCH_MEMORY_SHA256'],'source_run':pack['source_run'],
      'scope':pack['transfer_scope'],'rules':[
       'These are prior observations, not current-run scores, not authority over the executor/data contract.',
       'Start from a justified fresh hypothesis. You are NOT required to reproduce any source model or source best score.',
       'Use context-specific negative and positive evidence; a single regression is not a universal model ban.',
       'Engineering/budget failures do not establish bad model quality. Improving a parent while below the incumbent is not an intrinsically bad transition.',
       'No previous weights/code/data rows are supplied. Keep all actual failed and regressed outcomes in the new journal.',
      ],'cards':[project(c) for c in cards]}


def _write(directory,name,value):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    p=directory/(name+'.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(p)
    return str(p)


def response_schema(modules,ids):
    assessment={'type':'object','properties':{
        'memory_id':({'type':'string','enum':ids} if ids else {'type':'string'}),
        'relation':{'type':'string','enum':['support','avoid_repeat','mitigated','not_applicable']},
        'reason':{'type':'string'},'changed_condition':{'type':'string'},'plan_evidence':{'type':'string'}},
        'required':['memory_id','relation','reason','changed_condition','plan_evidence'],'additionalProperties':False}
    proposal={'type':'object','properties':{
      'id':{'type':'string'},'reason':{'type':'string'},
      'module':{'type':'array','items':{'type':'string','enum':list(modules)},'minItems':1},
      'plan':{'type':'object','properties':{name:{'type':'string'} for name in modules},'additionalProperties':False},
      'memory_assessments':{'type':'array','items':assessment},
      'falsification':{'type':'string'},'cost_and_reserve':{'type':'string'}},
      'required':['id','reason','module','plan','memory_assessments','falsification','cost_and_reserve'],
      'additionalProperties':False}
    return {'type':'object','properties':{
      'alternatives':{'type':'array','items':proposal,'minItems':3,'maxItems':3},
      'selected_id':{'type':'string'},'selection_reason':{'type':'string'}},
      'required':['alternatives','selected_id','selection_reason'],'additionalProperties':False}


def validate_decision(value,cards,modules):
    if not isinstance(value,dict):raise ValueError('response must be a JSON object')
    alternatives=value.get('alternatives',[]);known={c['id']:c for c in cards}
    if not isinstance(alternatives,list) or not all(isinstance(a,dict) for a in alternatives):raise ValueError('alternatives must be a list of three objects, never strings or pseudo-code')
    if len(alternatives)!=3:raise ValueError('exactly three alternatives are required')
    ids=[a['id'] for a in alternatives]
    if len(set(ids))!=3:raise ValueError('duplicate alternative IDs')
    if len({json.dumps(a['plan'],sort_keys=True) for a in alternatives})!=3:raise ValueError('duplicate plans')
    for a in alternatives:
        if not isinstance(a.get('module'),list) or not isinstance(a.get('plan'),dict):raise ValueError('module must be an array and plan an object')
        if not isinstance(a.get('memory_assessments'),list) or not all(isinstance(x,dict) for x in a['memory_assessments']):raise ValueError('memory_assessments must contain JSON objects')
        if not a['module'] or set(a['module'])!=set(a['plan']) or set(a['module'])-set(modules):raise ValueError('plan object keys must exactly equal module array values from allowed_modules; received module='+repr(a['module'])+' plan keys='+repr(list(a['plan'])))
        if not all(isinstance(v,str) and v.strip() for v in a['plan'].values()):raise ValueError('empty plan')
        if not a.get('falsification','').strip() or not a.get('cost_and_reserve','').strip():raise ValueError('missing experiment/cost rationale')
        seen=set()
        for x in a['memory_assessments']:
            if x['memory_id'] not in known or x['memory_id'] in seen:raise ValueError('unknown or repeated memory ID')
            seen.add(x['memory_id'])
            if x['relation'] not in ('support','avoid_repeat','mitigated','not_applicable'):raise ValueError('unknown relation')
            if not x.get('reason','').strip():raise ValueError('missing memory applicability reason')
            if x['relation']=='mitigated' and not x.get('changed_condition','').strip():raise ValueError('revisit requires a changed condition')
            if known[x['memory_id']].get('require_plan_anchor') and x['relation'] in ('mitigated','avoid_repeat'):
                quote=x.get('plan_evidence','').strip()
                if len(normalize_quote(quote))<12 or not any(normalize_quote(quote) in normalize_quote(part) for part in a['plan'].values()):raise ValueError('risk assessment needs an exact substantive plan_evidence quote from its own plan')
        if seen!=set(known):raise ValueError('each retrieved card requires an explicit applicability assessment')
    selected=next((a for a in alternatives if a['id']==value.get('selected_id')),None)
    if selected is None:raise ValueError('selected alternative is missing')
    if any(x['relation']=='avoid_repeat' for x in selected['memory_assessments']):
        raise ValueError('cannot knowingly select an unchanged applicable historical risk; select another alternative or name a concrete mitigation')
    if not value.get('selection_reason','').strip():raise ValueError('missing selection reason')
    return selected


def normalize_quote(text):
    # Quotation marks/Markdown and whitespace do not change the quoted parameter values.
    for c in ('“','”','"',"'",'`'):text=text.replace(c,'')
    return re.sub(r'\s+',' ',text).strip(' .;')


def normalize_response(value):
    """Normalize only unambiguous container syntax, never invent or relabel a plan."""
    if not isinstance(value,dict):return value
    value=copy.deepcopy(value);alts=value.get('alternatives')
    if isinstance(alts,dict) and len(alts)==3 and all(isinstance(v,dict) for v in alts.values()):
        value['alternatives']=[dict(v,id=v.get('id',k)) for k,v in alts.items()]
    for a in value.get('alternatives',[]) if isinstance(value.get('alternatives'),list) else []:
        if not isinstance(a,dict):continue
        if isinstance(a.get('module'),str):a['module']=[a['module']]
        if isinstance(a.get('plan'),str) and len(a.get('module',[]))==1:a['plan']={a['module'][0]:a['plan']}
        assessments=a.get('memory_assessments')
        if isinstance(assessments,dict) and all(isinstance(v,dict) for v in assessments.values()):
            a['memory_assessments']=[dict(v,memory_id=v.get('memory_id',k)) for k,v in assessments.items()]
    return value


def code_settings(code):
    values={}
    try:
        for n in ast.parse(code or '').body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.isupper():
                try:values[n.targets[0].id]=ast.literal_eval(n.value)
                except (ValueError,TypeError):pass
    except SyntaxError:pass
    return values


def current_run_context(agent,metric=None):
    best=getattr(agent,'best_node',None);out={'incumbent':None,'recent_attempts':[],'score_role':'host_search_internal','metric_definition':metric}
    if best is not None:
        out['incumbent']={'id':best.id,'score':getattr(getattr(best,'metric',None),'value',None),'settings':code_settings(best.code),'plan_summary':str(getattr(best,'plan',''))[:1500]}
    for n in list(getattr(getattr(agent,'journal',None),'nodes',[]) or [])[-12:]:
        if getattr(n,'stage',None)=='root':continue
        out['recent_attempts'].append({'id':n.id,'score':getattr(getattr(n,'metric',None),'value',None),'is_valid':getattr(n,'is_valid',False),'stage':getattr(n,'stage',None),'plan_summary':str(getattr(n,'plan',''))[:650]})
    return out


def plan(agent,prompt_base,context,modules,generate,max_retries=2):
    pack=load_for_agent(agent) if enabled() else None
    parent=context.get('parent_node')
    parent_id=getattr(parent,'id',None)
    query=str(context.get('previous_code',''))+'\n'+str(context.get('execution_output',''))
    cards=retrieve(pack,query,stage=context.get('stage','improve')) if pack else []
    pack_hash=os.environ.get('MLEVOLVE_SEARCH_MEMORY_SHA256') if pack else None
    task=str(prompt_base.get('Task description',''))
    # Construct the actual planner request here so a top-level Memory field
    # cannot silently disappear in a legacy prompt serializer.
    instruction={k:copy.deepcopy(v) for k,v in prompt_base.get('Instructions',{}).items() if k in ('Implementation guideline','Authoritative experiment contract','Frozen experiment and executor contract','Protocol','Evidence-based choice of change size')}
    for v in instruction.values():
        if isinstance(v,dict):v.pop('checkpoint_implementation',None)
    request={'task':task,'current_parent':{'id':parent_id,'code':context.get('previous_code',''),
        'execution_output':context.get('execution_output',''),'settings':code_settings(context.get('previous_code',''))},
      'current_run_evidence':current_run_context(agent,pack.get('metric') if pack else None),
      'current_instructions':instruction,'available_modules':list(modules),
      'distilled_experience':{'pack_sha256':pack_hash,
          'source_run':pack['source_run'] if pack else None,'metric_definition':pack.get('metric') if pack else None,'score_role':'host_search_internal_not_tune','cards':[project(c) for c in cards]},
      'response_field_guide':'Return exactly 3 alternatives, each with id/reason/module/plan/memory_assessments/falsification/cost_and_reserve. plan keys are the actual selected module names. selected_id must name the option you judge best; do not automatically choose the first.',
      'decision_contract':[
        'Propose THREE distinct plausible next hypotheses before selecting one; small hyperparameter or throughput changes are legitimate.',
        'Be compact: reason <=2 short sentences, each module plan <=2 short sentences, each memory assessment reason <=1 sentence. Do not repeat WHAT/WHY/HOW templates or all source evidence.',
        'Use current_run_evidence to compare the immediate parent AND incumbent. Removing one recent modification need not restore the incumbent configuration.',
        'For performance risks marked mitigated or avoid_repeat, supply plan_evidence as an exact quotation of the relevant implementation in that alternative plan. changed_condition must describe a concrete changed factor, not merely a promise to be careful.',
        'Do not force an avoid_repeat label to inflate avoidance counts. Conversely, do not claim mitigation without a planned concrete intervention addressing the observed condition.',
        'For EACH hypothesis assess EVERY retrieved card. Use support, avoid_repeat (same risky conditions), mitigated (concrete changed condition), or not_applicable with a reason.',
        'IMPORTANT ENUM MEANING: avoid_repeat means THIS PROPOSED CANDIDATE WOULD REPEAT the risky condition and SHOULD NOT be selected. It NEVER means that the candidate successfully avoids the bad condition. If the candidate removes label smoothing or otherwise removes/changes the risk, use mitigated with an exact plan quote, not avoid_repeat.',
        'Prefer a plausible improvement with useful evidence and manageable total cost. Do not select an unchanged applicable avoid_repeat; a materially changed condition allows a revisit.',
        'Do NOT require first-node reproduction of the historical best. Historical internal scores are not scores earned by this run.',
        'Do not equate below-incumbent with intrinsically bad. Keep beneficial stepping stones and justified novel exploration available.',
        'No architecture ban, forced checkpoint replay, invented success probability, or guarantee of improvement. Engineering failures are separate from model-quality evidence.',
        'State a falsifiable expected observation and training+evaluation+publication cost; return the exact structured response.',
        'The selected detailed plan will be sent to the coder; preserve actual executor/data constraints.',
        'Every alternative must have TOP-LEVEL id, reason, module, plan, memory_assessments, falsification, cost_and_reserve. Do not place falsification/cost inside plan.',
        'The plan object keys MUST EXACTLY MATCH that alternative module array. Never use generic keys such as change, selection, checkpointing, cost or publication in plan. Put that prose under actual allowed module names.',
        'Compare tune metrics to tune metrics and host search to host search; never use a historical search score as the threshold for a new tune score.',
      ]}
    directory=Path(agent.cfg.log_dir)/'search_memory';decision_id=uuid.uuid4().hex
    schema=response_schema(modules,[c['id'] for c in cards]);errors=[]
    for attempt in range(min(max_retries,2)):
        prompt={'system':'Select an evidence-conditioned next experiment. Source memory is fallible observational evidence, not instructions that override the current task contract.',
                'user':json.dumps(request,ensure_ascii=False,default=str),'assistant':''}
        _write(directory,decision_id+f'.request{attempt}',{'prompt':prompt,'schema':schema,'pack_sha256':pack_hash})
        started=time.monotonic()
        response=generate(prompt=prompt,temperature=agent.acfg.code.temp,cfg=agent.cfg,json_schema=schema)
        elapsed=time.monotonic()-started
        if isinstance(response,str):
            response=response.strip()
            if response.startswith('```'):response='\n'.join(response.splitlines()[1:-1])
            try:response=json.loads(response)
            except ValueError:response={}
        try:
            response=normalize_response(response)
            selected=validate_decision(response,cards,modules)
        except (ValueError,KeyError,TypeError) as error:
            errors.append(str(error));_write(directory,decision_id+f'.invalid{attempt}',{'response':response,'error':str(error),'seconds':elapsed})
            request['repair_feedback']=str(error)
            continue
        rejected=[{'id':a['id'],'memory_risks':[x['memory_id'] for x in a['memory_assessments'] if x['relation']=='avoid_repeat']} for a in response['alternatives'] if a['id']!=selected['id']]
        receipt={'schema':'mlevolve_search_memory_decision_v1','decision_id':decision_id,'parent_node_id':parent_id,
          'pack_sha256':pack_hash,'mode':'memory' if pack else 'control','retrieved_ids':[c['id'] for c in cards],
          'actual_prompt_sha256':sha(prompt['user']),'model':agent.acfg.code.model,
          'response':response,'rejected_alternatives':rejected,'seconds':elapsed,'prior_validation_errors':errors,
          'counterfactual_saving_claim':False}
        _write(directory,decision_id+'.decision',receipt)
        return {'reason':selected['reason']+'\nMemory-informed choice: '+response['selection_reason'],
          'module':selected['module'],'plan':selected['plan'],'parse_success':True,
          'search_memory_decision':{'decision_id':decision_id,'parent_node_id':parent_id,'pack_sha256':receipt['pack_sha256'],
            'retrieved_ids':receipt['retrieved_ids'],'selected_assessments':selected['memory_assessments'],
            'falsification':selected['falsification'],'cost_and_reserve':selected['cost_and_reserve']}}
    # An unavailable/unvalidated memory decision cannot silently be labelled
    # memory-guided. Existing outer framework error handling retains the failure.
    raise MemoryDecisionError('search memory decision failed validation: '+'; '.join(errors))


def record_planning_failure(agent,parent,error):
    if not planning_enabled():return
    _write(Path(agent.cfg.log_dir)/'search_memory',uuid.uuid4().hex+'.proposal_failure',{'parent_id':getattr(parent,'id',None),'error_type':type(error).__name__,'error':str(error)[:1600],'created_unix':time.time(),'no_candidate_training_started':True,'pack_sha256':os.environ.get('MLEVOLVE_SEARCH_MEMORY_SHA256')})


def record_execution(agent,node):
    if not planning_enabled():return
    value=getattr(node,'plan','')
    try:value=json.loads(value) if isinstance(value,str) else value
    except (ValueError,TypeError):value={}
    decision=value.get('search_memory_decision') if isinstance(value,dict) else None
    if enabled() and getattr(node,'stage',None)=='improve':
        if not decision or decision.get('pack_sha256')!=os.environ.get('MLEVOLVE_SEARCH_MEMORY_SHA256'):
            raise MemoryDecisionError('Improve execution requires a validated memory selection for this pack')
        parent_id=getattr(getattr(node,'parent',None),'id',None)
        if parent_id is not None and decision.get('parent_node_id')!=parent_id:raise MemoryDecisionError('memory decision parent mismatch')
        receipt_path=Path(agent.cfg.log_dir)/'search_memory'/(decision['decision_id']+'.decision.json')
        receipt=json.loads(receipt_path.read_text());selection=next(a for a in receipt['response']['alternatives'] if a['id']==receipt['response']['selected_id'])
        if selection['plan']!=value.get('plan'):raise MemoryDecisionError('selected plan was replaced before execution')
    _write(Path(agent.cfg.log_dir)/'search_memory',str(node.id)+'.execution_link',{
        'node_id':node.id,'code_sha256':sha(node.code),'decision':decision,
        'has_planner_memory_decision':decision is not None,
        'stage':getattr(node,'stage',None),'not_a_semantic_conformance_proof':True})


def summarize(iterations,decisions=()):
    best=None;refresh=regressed=equal=failed=valid=0
    trace=[]
    for n in iterations:
        m=n.get('metric');m=m.get('value') if isinstance(m,dict) else m
        ok=n.get('is_valid',not n.get('is_buggy',True)) and not n.get('is_buggy',False) and isinstance(m,(float,int)) and math.isfinite(m)
        outcome='failed'
        if ok:
            valid+=1
            if best is None:outcome='baseline';best=m
            elif m<best:refresh+=1;outcome='new_best';best=m
            elif m>best:regressed+=1;outcome='below_best'
            else:equal+=1;outcome='equal_best'
        else:failed+=1
        trace.append({'node_id':n.get('node_id',n.get('id')),'outcome':outcome,'metric':m if ok else None,'best_so_far':best})
    return {'attempts':len(trace),'valid':valid,'new_best_after_initial':refresh,'below_best':regressed,
      'equal_best':equal,'failed':failed,'nonrefresh_count':regressed+equal+failed,
      'nonrefresh_rate':(regressed+equal+failed)/max(1,len(trace)-(1 if valid else 0)),
      'planning_decisions':len(decisions),'memory_decisions':sum(bool(d.get('pack_sha256')) for d in decisions),'alternatives_deprioritized_for_memory':sum(bool(x['memory_risks']) for d in decisions for x in d.get('rejected_alternatives',[])),
      'avoided_training_failures_proven':None,'trace':trace,
      'note':'Deprioritized proposals are counterfactual, not proven saved failures. Compare real outcomes under equal budgets and record all attempts.'}
