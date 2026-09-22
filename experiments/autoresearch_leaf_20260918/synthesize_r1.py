"""Analyst-authored conditional synthesis, not a teacher-model call.
Interpretations below were read against the original journal and successful code.
"""
import json
from pathlib import Path
from core import atomic_json
ROOT=Path(__file__).resolve().parent
D=ROOT/'r1-distillation-v001'
E=json.loads((D/'evidence.json').read_text())
DEBUG={
1:'The first candidate failed before any scored checkpoint: it required schema input_columns to have length 192 although the ABI includes id. This is a verified runtime schema error, not evidence that CatBoost has poor accuracy. N2 repaired the predictor feature selection while retaining the full ABI frame and subsequently passed genuine reload and Host scoring.',
3:'The proposed ensemble was rejected before training for alleged class-order and partial-ensemble metadata risks. No measured ensemble score exists. The scientific effect of diversity is unknown; persist the actually fitted member list and integer-to-species mapping, then test a bounded ensemble if its total cost is justified.',
4:'The reviewer rejected this candidate while its own narrative states no concrete contract violation was established. There was no execution and hence no negative quality observation. In the new single-Agent harness, use executable schema/reload checks rather than importing this old reviewer veto as an architecture ban.',
5:'This pre-execution rejection contains contradictory claims about a no-fit fallback and dataframe extras, then acknowledges the generated checkpoint is consistent. There was no training score. Preserve exact input alignment and validate the actual artifact, but do not infer a failed model from the old review narrative.',
6:'This rejected review mixes incompatible interpretations of input_columns and claims removing id can reduce a 192-feature-only list below 192. No runtime evidence supports that claim. Inspect the actual frozen schema and derive predictors by excluding id/species, as the already certified N2 did; do not propagate the ambiguous reviewer story as fact.',
8:'A multi-seed proposal was never executed; the review oscillates between alleged partial-ensemble defects and acknowledging allowed behavior. No measured benefit or harm from seeds is available. A future ensemble should publish exactly its trained members and account for full refit cost, but this historical rejection is not a scientific veto.',
9:'The rejected candidate reviewer concludes that class alignment and fallback publication are acceptable and that status should be clean. This is an unexecuted review-control failure, not a model result. Keep deterministic executable tests and retain the last verified checkpoint if a future alternative fails.',
10:'The review identifies a conditional publication after a final usable fit as a potential deadline bug, alongside unverified class-order allegations. This candidate was not run, so the alleged failure is not empirically validated. A robust new implementation should always attempt durable publication after completed useful fitting and reserve reload/Host time.',
11:'This ensemble candidate was rejected although the review later says its single-member and ensemble save/reload paths are consistent and tune-only selection is allowed. No ensemble was measured. Treat it as a historical review inconsistency and distinguish selection on tune from prohibited fitting of model parameters on tune.',
12:'The reviewer alleges that stringified integer model classes were compared against species names, which would reject a checkpoint. The candidate was not executed, so this remains a static allegation rather than an observed failure. New predictors should explicitly map trained integer labels to frozen species columns and test the saved artifact.',
13:'Temperature calibration was proposed but unexecuted; the review admits it cannot establish a definitive ABI or leakage blocker. There is no calibration score or universal conclusion. Tune-only temperature selection is allowed under the current task, while feature/model fitting remains fit-only; preserve an uncalibrated fallback if it does not help.',
14:'Fit-only out-of-fold calibration was proposed but rejected before training; most reviewer claims are tentative class-representation concerns. Its performance is unknown and folds add substantial refitting cost. If revisited, budget all folds plus final fit and genuine replay, and explicitly map class indices rather than trusting label strings.'}
nodes={}
for n in E['nodes']:
    nid=n['node_id'];step=n['step']
    syn={'alternative':'Choose a different affordable model or representation if supported by current-run evidence; these observations neither require CatBoost reproduction nor prohibit images, neural models, calibration or ensembles.',
         'match_terms':['leaf','CatBoost','schema','checkpoint','budget']}
    if step in DEBUG:syn['debug_lesson']=DEBUG[step]
    else:
        syn['architecture_lesson']='A fit-only median-imputed 192-feature CatBoost MultiClass model produced a genuine reloadable 99-class checkpoint and independent Host score. This establishes feasibility of numerical descriptors, not superiority to image/neural alternatives. Preserve id only for alignment, reuse fit medians, and remap saved integer classes into frozen species order. The historical data split seed was 0 but this candidate hard-coded model seed 42; current runs use their own declared seed and resource cap.'
        syn['tuning_lesson']=(
            'N2 rebuilt fresh models at 300,600,900,1200 iterations. Tune loss decreased 2.118614→1.213122→0.851683→0.669105; final Host search was 0.716400. These are four fixed training stages, not four Agent tuning proposals, and no checkpoint continuation occurred. Keep learning_rate0.035,depth6,l2_leaf_reg8 semantics distinct from iteration count; account for the SUM of four fresh fits, not just 1200 trees. Other parameter optima are unmeasured.' if step==2 else
            'N7 differs from certified N2 only by extending stage_schedule to 1600,2000,2400 after the same 300,600,900,1200 stages. Later tune losses were 0.535736,0.454657,0.401943, with final independent Host search0.443894 versus N2 0.716400. This supports additional boosting capacity under the observed conditions, not a global optimum or cross-framework memory effect. All seven stages fresh-fit, so repeated early stages waste compute; continuation would be a NEW untested implementation requiring replay validation, not an observed source dependency.')
        ref=n['source_code']
        syn['mechanism_signature']={
            'preprocessing':'Exclude id/species; numeric coercion; infinities→NaN; fit medians only then fill0; float32; identical medians on tune/search/test; no scaling.',
            'initialization':'Fresh CatBoostClassifier per stage; no init_model or trained source checkpoint; random_seed42 in source.',
            'loss':'CatBoost MultiClass multiclass negative log-likelihood; Newton leaf estimation; l2_leaf_reg8 regularizes leaf estimation, not probability temperature.',
            'optimizer':'Boosting learning_rate0.035 per tree, depth6, random_strength0.7,border_count128,Bayesian bootstrap bagging_temperature0.5; CPU thread_count8 historical only.',
            'scheduler':'No learning-rate scheduler; iterations is total tree cap for EACH fresh fit, not incremental trees.',
            'checkpoint':'Save CatBoost model plus classes/index mapping/features/fit medians; fresh loader maps integer classes into schema order; probabilities clipped1e-12 and normalized. Host uses its separate original loss semantics.',
            'continuation_dependencies':'None reused across training stages; generic continuation boilerplate in pack does not describe these source stages.'}
        syn['mechanism_evidence']=[{'path':ref['source_artifact'],'sha256':ref['source_artifact_sha256'],'lines':[[10,64],[83,122],[130,200],[203,236]]}]
    nodes[nid]=syn
ids={n['step']:n['node_id'] for n in E['nodes']}
themes=[{'id':'r1_full_cost_and_unmeasured_alternatives','granularity':'tuning','node_ids':[ids[2],ids[7]],
 'title':'More boosting helped, but repeated fresh fits consumed the budget',
 'lesson':'The only two certified candidates share a numerical CatBoost recipe; N7 only extends iteration caps and improves Host search from0.716400 to0.443894. Execution cost rose from716.67 to2109.55 seconds (use exact source execution fields if estimates differ); node elapsed event timestamps also include non-training overhead. Eleven fresh-fit stage measurements are not eleven independent hypotheses. No image, neural, calibration or ensemble alternative earned a score, so changing family remains open. New4CPU resources differ from historical8CPU; remeasure throughput before selecting a cap.',
 'alternative':'Try fewer well-chosen fresh-fit caps or a genuinely validated continuation scheme, or evaluate another affordable family; retain durable publication and Host validation.',
 'match_terms':['CatBoost','iterations','fresh fit','cost','throughput']},
 {'id':'r1_review_failures_not_scientific_negatives','granularity':'debug','node_ids':[ids[x] for x in DEBUG],
 'title':'One executed schema failure, eleven unexecuted review rejections',
 'lesson':'The record contains one verified pre-checkpoint schema exception and eleven later pre-execution review rejections, several explicitly self-contradictory. None is a zero score. N2 repairs the first ABI issue and is Host-certified; the other rejected designs remain untested. The new harness does not contain the historical three-alternative reviewer, so its control-specific rules should not be reintroduced through memory.',
 'alternative':'Use real negative schema tests and fresh-process checkpoint replay; record runtime failures separately from rejected proposals and measured model regressions.',
 'match_terms':['review','ABI','schema','no score','failure'] }]
# Keep exact cost sourced rather than hand-rounded in scientific text.
themes[0]['lesson']=themes[0]['lesson'].replace('2109.55',f"{next(n for n in E['nodes'] if n['step']==7)['execution_seconds']:.2f}")
atomic_json(D/'analyst_synthesis.json',{'method':'analyst_conditional_trajectory_synthesis','nodes':nodes,'themes':themes})
print('Wrote analyst synthesis for',len(nodes),'nodes')
