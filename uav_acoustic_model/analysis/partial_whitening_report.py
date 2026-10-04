"""Offline truth labels, development-only selection and frozen evaluation."""
from collections import Counter
import json
import subprocess
import numpy as np
from analysis.partial_whitening_study import (OUT,ROOT,METHODS,BETAS,load,run_dir,beta_rows,
    verify_completion,audio_context,selection_manifest,replay_dir,replay_context)
from analysis.localization_error_attribution import _measurement_rows,_read_csv_gz,_write_csv_gz,_write_json
from validation.gazebo_experiment import sha256,canonical_sha256


def stats(rows):
    rows=sorted(rows,key=lambda r:r['frame_index'])
    valid=np.array([r['valid'] for r in rows]);errors=np.array([r['geodesic_error_deg'] for r in rows]);n=len(rows)
    if not n:raise ValueError('planned bearing cell is empty')
    bad=valid&(errors>10);lost=~valid
    result=dict(planned=n,invalid_count=int(lost.sum()),valid_count=int(valid.sum()),large_count=int(bad.sum()),
        loss=float((lost.sum()+bad.sum())/n),valid_fraction=float(valid.mean()),
        median_deg=float(np.median(errors[valid])) if valid.any() else None,
        p95_deg=float(np.percentile(errors[valid],95)) if valid.any() else None,
        invalid_reasons_json=json.dumps(dict(Counter(r['invalid_reason'] for r in rows if not r['valid'])),sort_keys=True))
    for threshold in (5,10,30):
        count=int((valid&(errors>threshold)).sum())
        result[f'over_{threshold}_count']=count
        result[f'over_{threshold}_all_planned']=count/n
        result[f'over_{threshold}_conditional']=count/valid.sum() if valid.any() else None
    for label,mask in [('large_error',bad),('invalid_or_large',lost|bad)]:
        durations=[];start=None;end=None
        for r,b in zip(rows,mask):
            if b:
                if start is None:start=r['frame_start_reception_time_s']
                end=r['frame_end_reception_time_s']
            elif start is not None:durations.append(end-start);start=None
        if start is not None:durations.append(end-start)
        result[label+'_series_count']=len(durations)
        result[label+'_series_max_s']=max(durations,default=0)
        result[label+'_series_total_s']=sum(durations)
        result[label+'_series_durations_s_json']=json.dumps(durations)
    return result


def cell_metrics(output,split):
    m=load(output);out=[];stride=m['processing']['tracker']['frame_stride']
    for spec in m['specs']:
        if spec['split']!=split:continue
        directory=run_dir(output,spec['index'])
        if not (directory/'completion.json').exists():continue
        verify_completion(directory,audio_context(output,spec['index']))
        rows=_measurement_rows(directory/'bearing_records.csv.gz')
        for method in METHODS:
            for beta in sorted({float(r['frontend_beta']) for r in rows if r['estimator_variant']==method},reverse=True):
                for station in ('S0','S1','S2'):
                    group=[r for r in rows if r['estimator_variant']==method and float(r['frontend_beta'])==beta and r['station_id']==station]
                    for sampling in ('all','stride'):
                        subset=group if sampling=='all' else [r for r in group if r['frame_index']%stride==0]
                        out.append(dict(split=split,index=spec['index'],source_class=spec['source_class'],distance_m=spec['distance_m'],replicate=spec['replicate'],method=method,beta=beta,station_id=station,sampling=sampling,**stats(subset)))
    return out


def macro(cells,method,beta,cls,sampling='all'):
    selected=[c for c in cells if c['method']==method and c['beta']==beta and c['source_class']==cls and c['sampling']==sampling]
    if not selected:raise ValueError('missing class/variant cells')
    result=dict(method=method,beta=beta,source_class=cls,sampling=sampling,cells=len(selected))
    for key in ('loss','valid_fraction','median_deg','p95_deg','over_5_conditional','over_10_conditional','over_30_conditional'):
        values=[c[key] for c in selected]
        result[key]=float(np.mean(values)) if all(v is not None for v in values) else None
    return result


def constraints(broad,harm,base_broad,base_harm):
    limit=None if base_broad['p95_deg'] is None else base_broad['p95_deg']+max(.1*base_broad['p95_deg'],.1)
    return dict(broadband_loss_ok=broad['loss']<=base_broad['loss']+.01+1e-12,
        broadband_p95_ok=limit is not None and broad['p95_deg'] is not None and broad['p95_deg']<=limit+1e-12,
        harmonic_valid_ok=harm['valid_fraction']>=base_harm['valid_fraction']-.01-1e-12,
        broadband_p95_limit_deg=limit)


def select_beta(cells):
    chosen={};candidates=[]
    for method in METHODS:
        bb=macro(cells,method,1,'broadband');hh=macro(cells,method,1,'nonstationary_harmonic');feasible=[]
        for beta in BETAS:
            b=macro(cells,method,beta,'broadband');h=macro(cells,method,beta,'nonstationary_harmonic')
            checks=constraints(b,h,bb,hh)
            ok=beta==1 or all(checks[k] for k in ('broadband_loss_ok','broadband_p95_ok','harmonic_valid_ok'))
            row=dict(method=method,beta=beta,harmonic_loss=h['loss'],harmonic_valid_fraction=h['valid_fraction'],
                broadband_loss=b['loss'],broadband_p95_deg=b['p95_deg'],feasible=ok,**checks)
            candidates.append(row)
            if ok:feasible.append(row)
        minimum=min(r['harmonic_loss'] for r in feasible)
        tied=[r for r in feasible if abs(r['harmonic_loss']-minimum)<=1e-12]
        chosen[method]=max(r['beta'] for r in tied)
    return chosen,candidates


def freeze(output=OUT):
    m=load(output)
    if (output/'selection_manifest.json').exists():raise FileExistsError('beta already frozen')
    if any(run_dir(output,s['index']).exists() for s in m['specs'] if s['split']=='evaluation'):
        raise ValueError('evaluation accessed before selection freeze')
    cells=cell_metrics(output,'development')
    if len(cells)!=8*3*2*3*2:raise ValueError('all eight development streams and three betas required')
    selected,candidates=select_beta(cells)
    _write_csv_gz(output/'development_metrics.csv.gz',cells);_write_csv_gz(output/'development_tradeoff.csv.gz',candidates)
    files={p.relative_to(output).as_posix():sha256(p) for i in range(8) for p in run_dir(output,i).iterdir() if p.is_file()}
    for p in (output/'source_banks').glob('development*'):files[p.relative_to(output).as_posix()]=sha256(p)
    for name in ('development_metrics.csv.gz','development_tradeoff.csv.gz'):files[name]=sha256(output/name)
    g=dict(schema_version=1,status='frozen_before_evaluation',selected_beta=selected,source_sha256=m['source_sha256'],
        processing_code_sha256=m['processing_code_sha256'],study_manifest_sha256=sha256(output/'study_manifest.json'),
        development_sha256=files,selection_rule=m['selection_rule'],ambiguity_gate_enabled=False)
    _write_json(output/'selection_manifest.json',g)
    return selected


def verify(output=OUT):
    m=load(output);audio=0;replays=0;reuses=0
    ledger=json.loads((output/'execution_ledger.json').read_text())
    for a in ledger['attempts']:
        if a['status']=='complete':
            d=run_dir(output,a['index']) if a['kind']=='audio' else replay_dir(output,a['index'],a['method'],a['mode'])
            if sha256(d/'completion.json')!=a['completion_sha256']:raise ValueError('completion ledger SHA mismatch')
    for spec in m['specs']:
        d=run_dir(output,spec['index'])
        if (d/'completion.json').exists():verify_completion(d,audio_context(output,spec['index']));audio+=1
    if (output/'selection_manifest.json').exists():
        g=selection_manifest(output,m)
        for d in sorted((output/'replays').glob('*')):
            if d.is_dir():
                c=json.loads((d/'completion.json').read_text());verify_completion(d,replay_context(output,c['index'],c['method'],c['mode']))
                s=json.loads((d/'summary.json').read_text());beta=1 if c['mode']=='baseline' else g['selected_beta'][c['method']]
                if s['beta']!=beta or s['ambiguity_gate_enabled']:raise ValueError('replay frontend mismatch')
                replays+=1
            else:
                reuse=json.loads(d.read_text());base=output/reuse['baseline_path']
                if g['selected_beta'][reuse['method']]!=1 or reuse['baseline_completion_sha256']!=sha256(base/'completion.json') or reuse['selection_manifest_sha256']!=sha256(output/'selection_manifest.json'):
                    raise ValueError('baseline reuse mismatch')
                source=run_dir(output,reuse['index'])
                ids=[r['event_id'] for r in beta_rows(source,1.,reuse['method'])]
                if reuse['input_bearings_sha256']!=sha256(source/'bearing_records.csv.gz') or reuse['input_event_ids_sha256']!=canonical_sha256(ids):raise ValueError('reuse input SHA mismatch')
                verify_completion(base,replay_context(output,reuse['index'],reuse['method'],'baseline'));reuses+=1
    return dict(audio_complete=audio,replay_computations=replays,identical_beta_one_reuses=reuses)


def report(output=OUT):
    m=load(output);evidence=verify(output)
    dev=cell_metrics(output,'development');ev=cell_metrics(output,'evaluation')
    if dev:_write_csv_gz(output/'development_metrics.csv.gz',dev)
    if ev:_write_csv_gz(output/'evaluation_metrics.csv.gz',ev)
    selection=selection_manifest(output,m) if (output/'selection_manifest.json').exists() else None
    macros=[];criteria=[];tracking=[];pairs=[]
    if selection:
        for split,cells in [('development',dev),('evaluation',ev)]:
            for method in METHODS:
                for beta in (BETAS if split=='development' else sorted({1.,selection['selected_beta'][method]},reverse=True)):
                    for cls in ('broadband','nonstationary_harmonic'):
                        for sampling in ('all','stride'):
                            if cells:macros.append(dict(split=split,**macro(cells,method,beta,cls,sampling)))
        for d in sorted((output/'replays').glob('*')):
            if d.is_dir():tracking.append(json.loads((d/'summary.json').read_text()))
        for method in METHODS:
            beta=selection['selected_beta'][method];method_pairs=[]
            for cls in ('broadband','nonstationary_harmonic'):
                off=next((r for r in tracking if r['source_class']==cls and r['method']==method and r['mode']=='baseline'),None)
                on=off if beta==1 else next((r for r in tracking if r['source_class']==cls and r['method']==method and r['mode']=='selected'),None)
                if off is None or on is None:continue
                pair=dict(method=method,source_class=cls,beta=beta,selected_reuses_baseline=beta==1)
                keys=('ever_confirmed','first_confirmation_time_s_relative','first_confirmation_position_error_m','availability_fraction',
                    'position_rmse_m_conditional','position_p95_m_conditional','position_over_10m_all_epochs','position_over_50m_all_epochs',
                    'total_executed_optimizations','accepted_update_count','rejected_update_count','final_status','final_failure_reason')
                for mode,r in [('baseline',off),('selected',on)]:
                    pair.update({mode+'_'+k:r[k] for k in keys})
                pair['tracking_preservation_passed']=bool((not off['ever_confirmed'] or on['ever_confirmed']) and on['availability_fraction']>=off['availability_fraction']-.05-1e-12 and on['position_over_50m_all_epochs']<=off['position_over_50m_all_epochs']+1e-12)
                pairs.append(pair);method_pairs.append(pair)
            if ev:
                baseb=macro(ev,method,1,'broadband');baseh=macro(ev,method,1,'nonstationary_harmonic')
                broad=macro(ev,method,beta,'broadband');harm=macro(ev,method,beta,'nonstationary_harmonic')
                checks=constraints(broad,harm,baseb,baseh)
                reduction=(baseh['loss']-harm['loss'])/baseh['loss'] if baseh['loss'] else None
                passed=bool(evidence['audio_complete']==16 and reduction is not None and reduction>=.25 and all(checks[k] for k in ('broadband_loss_ok','broadband_p95_ok','harmonic_valid_ok')) and len(method_pairs)==2 and all(p['tracking_preservation_passed'] for p in method_pairs))
                criteria.append(dict(method=method,beta=beta,baseline_harmonic_loss=baseh['loss'],selected_harmonic_loss=harm['loss'],harmonic_loss_relative_reduction=reduction,promising_for_further_calibration=passed,**checks))
    _write_csv_gz(output/'macro_metrics.csv.gz',macros,columns=None if macros else ('split','method','beta','source_class','sampling','loss','valid_fraction'))
    if tracking:_write_csv_gz(output/'tracking_metrics.csv.gz',tracking)
    if pairs:_write_csv_gz(output/'paired_tracking_review.csv.gz',pairs)
    ledger=json.loads((output/'execution_ledger.json').read_text())
    result=dict(**evidence,worker_wall_s=ledger['worker_wall_s'],technical_stop=ledger['stopped'],stop_reason=ledger.get('stop_reason'),
        selected_beta=selection['selected_beta'] if selection else None,evaluation_criteria=criteria,
        matrix_planned_audio=16,matrix_planned_replay_comparisons=8,ambiguity_gate_enabled=False,baseline_R_reused=True,
        scope='known Gazebo trajectories; dependent synthetic transfer, not independent flights or iid frames')
    _write_json(output/'review.json',result)
    return result
