"""Saved-artifact evaluation of the predeclared success criteria, no tuning."""
from pathlib import Path
import json
import numpy as np
from analysis.bearing_ambiguity_study import load,gate_manifest,METHODS
from analysis.bearing_ambiguity_failure import audit_stop
from analysis.bearing_ambiguity_report import verify
from analysis.localization_error_attribution import _read_csv_gz,_write_json,_write_csv_gz


def evaluate_saved(output: Path):
    m=load(output);g=gate_manifest(output,m)
    audio,replays=verify(output)
    aggregate=json.loads((output/'aggregate.json').read_text())
    paired=[];bearing=[]
    macro=aggregate['evaluation_macro']
    for method in METHODS:
        off=next(r for r in macro if r['method']==method and r['gate']=='off')
        on=next(r for r in macro if r['method']==method and r['gate']=='on')
        baseline=off['over_10_passed_all_valid']
        reduction=(baseline-on['over_10_passed_all_valid'])/baseline if baseline else None
        good=on['good_false_rejection_conditional']
        passed=bool(reduction is not None and reduction>=.25 and on['retention']>=.7 and good is not None and good<=.05)
        bearing.append(dict(method=method,baseline_over_10_all_valid=baseline,gated_over_10_all_valid=on['over_10_passed_all_valid'],
            relative_reduction=reduction,retention=on['retention'],good_false_rejection=good,predeclared_bearing_criterion_passed=passed))
    summaries=[json.loads(p.read_text()) for p in sorted((output/'replays').glob('*/summary.json'))]
    for cls in ('broadband','nonstationary_harmonic'):
        for method in METHODS:
            found=[r for r in summaries if r['source_class']==cls and r['method']==method]
            off=next((r for r in found if r['gate']=='off'),None);on=next((r for r in found if r['gate']=='on'),None)
            if off is None or on is None:
                paired.append(dict(source_class=cls,method=method,complete_pair=False));continue
            pair=dict(source_class=cls,method=method,complete_pair=True)
            for mode,r in [('off',off),('on',on)]:
                for key in ('ever_confirmed','first_confirmation_time_s_relative','first_confirmation_position_error_m',
                    'availability_fraction','position_rmse_m_conditional','position_p95_m_conditional',
                    'position_over_10m_all_epochs','position_over_50m_all_epochs','total_executed_optimizations',
                    'accepted_update_count','rejected_update_count','final_status','final_failure_reason','gate_rejection_count'):
                    pair[mode+'_'+key]=r[key]
            pair['no_confirmation_loss']=not off['ever_confirmed'] or on['ever_confirmed']
            pair['availability_loss']=off['availability_fraction']-on['availability_fraction']
            pair['tail_50m_change']=on['position_over_50m_all_epochs']-off['position_over_50m_all_epochs']
            pair['predeclared_tracking_preservation_passed']=bool(pair['no_confirmation_loss'] and pair['availability_loss']<=.05 and pair['tail_50m_change']<=0)
            paired.append(pair)
    success=audio==16 and replays==8 and all(b['predeclared_bearing_criterion_passed'] for b in bearing) and all(p.get('predeclared_tracking_preservation_passed',False) for p in paired)
    _write_csv_gz(output/'paired_tracking_review.csv.gz',paired)
    previous=audit_stop()
    result=dict(audio_complete=audio,audio_planned=16,replays_complete=replays,replays_planned=8,
        worker_wall_s=aggregate['worker_wall_s'],previous_worker_wall_s=previous['worker_wall_s'],
        bearing=bearing,tracking=paired,predeclared_joint_criterion_passed=success,
        feature_thresholds=g['thresholds'],development_selection_failed=g['development_selection_failed'],
        scientific_scope='dependent synthetic transfer; no iid-frame or population claim',gate_default_enabled=False)
    _write_json(output/'review.json',result)
    return result
