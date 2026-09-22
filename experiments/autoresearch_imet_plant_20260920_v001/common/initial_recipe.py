"""One-off memory-anchored initial proposal audit; no restriction after valid incumbent."""
import hashlib,json,re
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parent.parent/'contract/r2_recipe_contract.json'
R2_RECIPE_CONTRACT = json.loads(CONTRACT_PATH.read_text())
MANDATORY_SOURCE_PATCH = r'''Mandatory executable fragments: the generated Python source must contain exactly one backslash character in `words=re.findall(r'\b\w+\b', text, flags=re.UNICODE)` (two consecutive backslashes in the source are forbidden). The NB block must contain `presence=CountVectorizer(binary=True,ngram_range=(1,2),min_df=1,max_features=180000)`, `X=presence.fit_transform(texts); d=X.shape[1]`, feature vectors `pos=np.asarray(X[y==j].sum(axis=0)).ravel()` and `neg=np.asarray(X[y!=j].sum(axis=0)).ravel()`, then the exact `ratio=np.log((pos/(pos.sum()+d)+1/d)/(neg/(neg.sum()+d)+1/d))`; fit one one-vs-rest model per class with `X.multiply(ratio)` or `X.multiply(ratios[j])` and never use ratio.mean(), r.mean(), pos.sum(1), neg.sum(1), or a single multiclass NB model. Prediction must column-stack three positive-class probabilities and row-normalize. Include `sentence_group_count=max(1,len(re.findall(r'[.!?]+', text)))` (or an exact equivalent) in the style features. The final memory_decisions keys must equal selected_memory_keys exactly, once each; never invent keys.'''
REQUIREMENTS = '''This revised rerun starts from a memory-anchored FIRST proposal, not the common weak baseline. Read authorized R1/R2 architecture+tuning cards to reconstruct a strongly supported compatible recipe, fresh-fit all weights. Explain why the retrieved recipe is a useful starting point. Prioritize the six-view three-fold OOF sparse stack with a genuinely case-preserving word branch, the recorded NB log-ratio semantics and stylometry; use compatible historically successful meta regularization/temperature anchors and a small tune-only neighborhood. Do not silently replace it with a cheaper weak baseline. Parameter meanings and units must match the retrieved card and the attached R2 recipe contract. In particular, the NB branch must use d=presence_matrix.shape[1] in ratio = log((pos/(sum(pos)+d)+1/d)/(neg/(sum(neg)+d)+1/d)); positive-class NB probabilities must be column-stacked in EAP/HPL/MWS order and row-normalized before log features. Fold preprocessing/base weights are fit on each training fold only; meta weights use assembled fit OOF features; choose meta-C and temperature by real tune predictions, never meta training residual or a placeholder score. Generic future pretrained/novel routes remain allowed after this first successful candidate. No score reproduction requirement. Return architecture, selected parameters, explicit data roles, and each internal evaluation with score_role in train metadata.'''
CHECKS = ['memory_recipe_grounded','actual_case_preservation','fit_only_fold_preprocessing','oof_meta_fit','tune_only_selection','parameter_semantics','nb_recipe_contract_static','nb_probability_normalization','style_feature_semantics','fresh_weights','host_labels_inaccessible']
class InitialRecipeError(ValueError):pass

def unwrap_revision(response,out,save):
 """Accept the Agent's explicit {candidate:{...}} envelope without hiding it."""
 if isinstance(response,dict) and isinstance(response.get('candidate'),dict) and 'code' in response['candidate']:
  candidate=response['candidate']
  save(out/'04-recipe-revision-1-unwrapped.json',{'wrapped_response_keys':sorted(response.keys()),'candidate_keys':sorted(candidate.keys()),'envelope_preserved_in_parsed_response':True})
  return candidate
 return response

def check_report(report,code,readids):
 errors=[]
 if report.get('candidate_sha256')!=hashlib.sha256(code.encode()).hexdigest():errors.append('candidate hash mismatch')
 if not report.get('memory_refs') or any(k not in readids for k in report['memory_refs']):errors.append('missing or unselected memory refs')
 if not report.get('rationale'):errors.append('missing scientific rationale')
 checks=report.get('checks',{})
 for key in CHECKS:
  v=checks.get(key,{})
  if v.get('passed') is not True:errors.append(key+': not verified')
  fragment=v.get('code_evidence')
  if not isinstance(fragment,str) or not fragment or fragment not in code:errors.append(key+': exact source evidence missing')
 # A report flag alone is not enough for the most failure-prone recipe contract.
 # These checks are static guards; execution and Host recomputation remain separate.
 if not re.search(r'd\s*=\s*X\.shape\[1\]',code):errors.append('nb_recipe_contract_static: missing d from presence vocabulary')
 if not (re.search(r'pos\s*/\s*\(pos\.sum\(\)\s*\+\s*d\)',code) and re.search(r'neg\s*/\s*\(neg\.sum\(\)\s*\+\s*d\)',code) and re.search(r'1(?:\.0)?\s*/\s*d',code)):errors.append('nb_recipe_contract_static: recorded d/1-d smoothing formula not found')
 if re.search(r'(?:pos|neg)\s*=.*\+\s*1e-12',code) or re.search(r'(?:pos|neg)\.sum\(\s*1\s*,\s*keepdims',code):errors.append('nb_recipe_contract_static: epsilon or row-sum semantics altered the recorded scalar formula')
 if not re.search(r'CountVectorizer\([^\n]*binary\s*=\s*True',code) or not re.search(r'presence\s*=\s*CountVectorizer',code):errors.append('nb_recipe_contract_static: NB must fit a distinct binary presence CountVectorizer')
 if "r'\\b\\w+\\b'" not in code and 'r"\\b\\w+\\b"' not in code:errors.append('style_feature_semantics: missing literal word-boundary regex')
 if 'ratio.mean()' in code or not (re.search(r'ratios?\s*\[\s*[^\]]+\s*\]',code) or re.search(r'X\.multiply\(ratio\)',code)):errors.append('nb_recipe_contract_static: NB must use a per-class ratio index or loop-local ratio, not a mean ratio')
 return errors

def audit_and_revise(response,readids,cards,out,deadline,ask,save,contract,repair):
 for n in range(3):
  code=response['code'];digest=hashlib.sha256(code.encode()).hexdigest()
  request={'candidate_sha256':digest,'code':code,'retrieved_memory':cards,'r2_recipe_contract':R2_RECIPE_CONTRACT,'allowed_memory_refs':readids,'requirements':REQUIREMENTS,'request':'Audit actual code/data flow, not comments. Return candidate_sha256, memory_refs using only the exact allowed_memory_refs (full channel-prefixed keys; do not include the audit contract as a memory ref), rationale and checks mapping every required check to {passed:bool,code_evidence:exact nonempty source substring,reason:string}. If uncertain mark false. Copy every code_evidence character-for-character from code; use short single-line snippets when possible and verify evidence in code before returning. For parameter_semantics, evidence must show the NB d/1-d formula, not merely a C or temperature grid. For oof_meta_fit and fresh_weights, use exact single-line evidence. OOF base features do not make meta training-residual selection honest tune selection. Verify explicit lowercase=False or equivalent actual behavior, NB formula with d taken from the actual presence vocabulary, positive-class NB probability row normalization before log features, recorded normalized stylometry semantics, fit-only fold preprocess, and independent meta-C/temperature tune selection. A low historic score is not proof of faithful code.','required_checks':CHECKS}
  request['request'] += '\n' + MANDATORY_SOURCE_PATCH
  report=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps(request,ensure_ascii=False)}],out/f'04-recipe-audit-{n}',deadline)
  errors=check_report(report,code,readids);save(out/f'04-recipe-audit-{n}-validation.json',{'errors':errors,'candidate_sha256':digest,'semantic_judgment':'Agent source audit, not proof of future score'})
  if not errors:return response,report
  if n==2:raise InitialRecipeError('initial recipe audit unresolved: '+'; '.join(errors))
  response=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps({'candidate':response,'selected_keys':readids,'retrieved_memory':cards,'r2_recipe_contract':R2_RECIPE_CONTRACT,'audit':report,'errors':errors,'request':'Revise only to resolve these initial-recipe semantic issues. Return full proposal code/change/budget_seconds/public_models/memory_decisions. Use the R2 recipe contract literally: fit a distinct binary presence vectorizer in the executable NB training block. The generated source MUST literally contain and execute `presence=CountVectorizer(binary=True,...)` followed by `X=presence.fit_transform(texts); d=X.shape[1]`; a helper such as `_vec()` that merely returns CountVectorizer does NOT satisfy this requirement and must not replace the explicit assignment. Keep the presence object separate from every TF-IDF vectorizer and persist it for prediction. Then use the scalar formula ratio = log((pos/(pos.sum()+d)+1/d)/(neg/(neg.sum()+d)+1/d)); train and predict each one-vs-rest NB model with its own per-class vector X.multiply(ratio) inside the class loop or X.multiply(ratios[j]) at prediction time; never use ratio.mean() or a single multiclass fit for this branch; do not reuse a TF-IDF X, add 1e-12 to pos/neg, or replace scalar sums with axis or row sums. Preserve p=column_stack(three positive-class probabilities) followed by row normalization before log features. For stylometry use exactly the executable Python source words=re.findall(r\"\\b\\w+\\b\", text, flags=re.UNICODE) with ONE backslash character before each letter in the generated file. Do not write r\"\\\\b\\\\w+\\\\b\", because that matches literal slashes rather than word boundaries. Copy the exact one-backslash source fragment into audit evidence. In the same style feature row, include the recorded normalized sentence-group count as np.log1p(sentence_group_count) or an explicitly equivalent sentence-count variable, along with log1p length/word-count, mean/std token length, distinct lowercase-word ratio, alphabetic/length, uppercase/alphabetic, digit/length, normalized punctuation and whitespace; do not substitute distinct-word count for sentence-group count. Keep real tune-only selection: loop over meta-C and temperature, compute log_loss on predictions of /data/tune.csv, and record score_role=\"tune_selection\"; do not use a fit residual or placeholder. Preserve every selected memory key exactly once in memory_decisions; selected keys are frozen and must not be replaced by other catalogue keys. For adopt/adapt entries, provide code_evidence copied character-for-character from the revised code (short exact substrings are safest; do not paraphrase or alter punctuation). If a selected card is not actually implemented, use reject or defer with no code_evidence. This is scientific repair; explain actual changes. Do not claim a historical score for untrained code.'+'\n'+MANDATORY_SOURCE_PATCH},ensure_ascii=False)}],out/'04-recipe-revision-1',deadline)
  response=unwrap_revision(response,out,save)
  try:
   compile(response['code'],'candidate.py','exec')
  except SyntaxError as exc:
   save(out/'04-recipe-syntax-error-1.json',{'stage':'revision-1','error':str(exc),'lineno':exc.lineno,'offset':exc.offset,'text':exc.text})
   retry_payload={'candidate':response,'selected_keys':readids,'retrieved_memory':cards,'syntax_error':str(exc),'request':'The revised candidate failed Python compilation. Repair syntax only while preserving the requested R2 recipe semantics and all selected memory decisions. Return a full proposal object or an explicit candidate envelope with complete code, change, budget_seconds, public_models, and memory_decisions. The returned source must compile with compile(code, candidate.py, exec) before any audit. Keep executable distinct NB presence assignment presence=CountVectorizer(binary=True,...), X=presence.fit_transform(texts), d=X.shape[1], the exact scalar pos.sum()+d and neg.sum()+d formula, positive-class probability column stacking and row normalization, and a real raw word-boundary regex r\'\\b\\w+\\b\'. Do not claim an untrained score.'}
   response=ask([{'role':'system','content':contract},{'role':'user','content':json.dumps(retry_payload,ensure_ascii=False)}],out/'04-recipe-syntax-repair-1',deadline)
   response=unwrap_revision(response,out,save)
   try:
    compile(response['code'],'candidate.py','exec')
   except SyntaxError as exc2:
    save(out/'04-recipe-syntax-error-2.json',{'stage':'syntax-repair-1','error':str(exc2),'lineno':exc2.lineno,'offset':exc2.offset,'text':exc2.text})
    raise InitialRecipeError('revision syntax unresolved: '+str(exc2))
  repair_out=out/f'04-reference-recheck-{n}';repair_out.mkdir(exist_ok=False)
  response,_=repair(response,readids,cards,repair_out,deadline,ask,save,contract)
