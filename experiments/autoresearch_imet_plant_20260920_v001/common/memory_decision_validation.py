"""Lossless selected-key normalization and code-bound decision-only repair."""
import copy,hashlib,json

class MemoryDecisionValidationError(ValueError):
    pass

def code_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()

def evidence_hints(code):
    """Return short substrings copied from the frozen candidate for the repair Agent.

    These hints do not relax validation: the returned evidence is still checked with
    the exact `substring in code` rule below. They reduce paraphrased snippets after
    a scientific revision has frozen the candidate.
    """
    out=[]
    for line in code.splitlines():
        s=line.strip()
        if 8 <= len(s) <= 220 and s not in out:
            out.append(s)
    return out[:160]

def validate(raw_decisions, selected_keys, code):
    normalized=[];mappings=[];errors=[]
    if len(selected_keys)!=len(set(selected_keys)):
        errors.append({'error':'duplicate selected keys'})
    if not isinstance(raw_decisions,list):
        return [],[],[{'error':'memory_decisions must be an array'}]
    selected=set(selected_keys)
    for index,item in enumerate(raw_decisions):
        if not isinstance(item,dict):
            errors.append({'index':index,'error':'decision must be an object'});continue
        raw=item.get('key');canonical=None
        if not isinstance(raw,str):
            errors.append({'index':index,'error':'key must be a string'});continue
        if raw in selected:
            canonical=raw;method='exact_selected_key'
        else:
            matches=[key for key in selected_keys if key.partition(':')[2]==raw]
            if len(matches)==1:
                canonical=matches[0];method='unique_bare_id_in_selected_set'
            else:
                errors.append({'index':index,'raw_key':raw,'error':'ambiguous_selected_id' if len(matches)>1 else 'unselected_or_unknown_id','matches':matches});continue
        d=copy.deepcopy(item);d['key']=canonical;normalized.append(d)
        mappings.append({'index':index,'raw_key':raw,'normalized_key':canonical,'method':method})
        if d.get('action') not in ['adopt','adapt','reject','defer']:
            errors.append({'key':canonical,'error':'invalid action'})
        if not isinstance(d.get('reason'),str) or not d['reason'].strip():
            errors.append({'key':canonical,'error':'nonempty reason required'})
        if d.get('action') in ['adopt','adapt']:
            evidence=d.get('code_evidence')
            if not isinstance(evidence,str) or not evidence or evidence not in code:
                errors.append({'key':canonical,'error':'code_evidence must be an exact nonempty substring of frozen code'})
    keys=[d['key'] for d in normalized]
    if set(keys)!=selected or len(keys)!=len(set(keys)):
        errors.append({'error':'selected coverage mismatch','missing':sorted(selected-set(keys)),'duplicate_keys':len(keys)!=len(set(keys))})
    return normalized,mappings,errors

def repair_decisions(proposal, selected_keys, cards, out, deadline, ask, save, contract):
    original=copy.deepcopy(proposal);code=original['code'];digest=code_hash(code)
    raw=copy.deepcopy(original.get('memory_decisions',[]));protocol_errors=[];history=[]
    for index in range(3):
        normalized,mapping,errors=validate(raw,selected_keys,code)
        errors=protocol_errors+errors
        item={'attempt':index,'candidate_sha256':digest,'raw_decisions':copy.deepcopy(raw),'normalized_decisions':normalized,'id_mapping':mapping,'errors':errors}
        history.append(item);save(out/f'03-memory-validation-{index}.json',item)
        if not errors:
            result=copy.deepcopy(original);result['memory_decisions']=normalized
            receipt={'schema':'selected-reference-code-bound-v1','candidate_sha256':digest,'candidate_code_unchanged':True,'repair_calls':index,'original_raw_decisions':original.get('memory_decisions',[]),'final_raw_decisions':raw,'normalized_decisions':normalized,'id_mapping':mapping,'history':history,'training_executed':False,'limits':'Exact code references are syntactic evidence, not proof of scientific implementation or benefit.'}
            save(out/'memory_reference_normalization.json',receipt)
            assert result['code']==original['code'] and code_hash(result['code'])==digest
            return result,receipt
        if index==2:
            save(out/'MEMORY_VALIDATION_FAILURE.json',{'failure_phase':'memory_validation','memory_validation_failed':True,'training_executed':False,'candidate_sha256':digest,'history':history})
            raise MemoryDecisionValidationError('memory decision evidence invalid after two decision-only corrections')
        prompt={'candidate_sha256':digest,'frozen_candidate_code':code,'verbatim_evidence_hints':evidence_hints(code),'selected_keys':selected_keys,'historical_memory':cards,'previous_decisions':raw,'validation_errors':errors,'request':'Return ONLY {"candidate_sha256": the exact provided hash, "memory_decisions":[{"key": exact selected key INCLUDING its channel prefix, "action":"adopt|adapt|reject|defer", "reason":nonempty rationale, "code_evidence":exact verbatim substring of frozen_candidate_code for adopt/adapt}]}. The only legal key strings are exactly this JSON array and no others: '+json.dumps(selected_keys,ensure_ascii=False)+'. Include every one of these keys exactly once, and include no key inferred from card prose, previous decisions, or related cards. If a key is not in that exact array, delete it. For adopt/adapt, copy one string from verbatim_evidence_hints character-for-character, including punctuation and arguments; do not shorten, retype, normalize, or paraphrase it. Reject/defer a card honestly if no actual implementation supports adoption. Correct references/evidence only. Candidate code, architecture, budget, parameters and change description are frozen. Do NOT return code or any other fields. Validation remains exact substring matching.'}
        patch=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],out/f'03-decision-correct-{index+1}',deadline)
        protocol_errors=[]
        if not isinstance(patch,dict) or set(patch)!={'candidate_sha256','memory_decisions'}:
            protocol_errors=[{'error':'decision patch must contain only candidate_sha256 and memory_decisions; code regeneration is prohibited'}]
        elif patch['candidate_sha256']!=digest:
            protocol_errors=[{'error':'candidate_sha256 differs from frozen candidate','expected':digest}]
        else:
            raw=copy.deepcopy(patch['memory_decisions'])
    raise AssertionError('unreachable')
