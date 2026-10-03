"""Offline labels and reporting live here, outside the truth-free gate."""
from collections import Counter, deque
from pathlib import Path
import json
import numpy as np
from analysis.bearing_ambiguity_gate import GateConfig,decision
from analysis.bearing_ambiguity_study import (OUT,ROOT,METHODS,load,run_dir,read_features,verify_completion,gate_manifest)
from analysis.localization_error_attribution import _measurement_rows,_write_csv_gz,_write_json,_read_csv_gz
from validation.gazebo_experiment import sha256


def cells(output,split):
    m=load(output);result=[]
    for spec in m['specs']:
        if spec['split']!=split:continue
        directory=run_dir(output,spec['index'])
        if not (directory/'completion.json').exists():continue
        verify_completion(directory,{'index':spec['index'],'kind':'audio','study_manifest_sha256':sha256(output/'study_manifest.json')})
        rows=_measurement_rows(directory/'bearing_records.csv.gz');features=read_features(directory)
        for method in METHODS:
            for station in ('S0','S1','S2'):
                group=[r for r in rows if r['estimator_variant']==method and r['station_id']==station]
                result.append((spec,method,station,group,features))
    return result


def stats(rows,features,config):
    valid=np.array([r['valid'] for r in rows],bool)
    accepted=np.array([r['valid'] and decision(features[r['event_id']],config)[0] for r in rows],bool)
    errors=np.array([r['geodesic_error_deg'] for r in rows]);n=int(valid.sum())
    good=valid&(errors<=5)
    result=dict(original_frames=len(rows),original_valid=n,accepted=int(accepted.sum()),rejected=int((valid&~accepted).sum()),
        accepted_fraction_all_frames=float(accepted.mean()),retention=float(accepted.sum()/n) if n else None,
        good_count=int(good.sum()),good_false_rejection_count=int((good&~accepted).sum()),
        good_false_rejection_conditional=float((good&~accepted).sum()/good.sum()) if good.any() else None,
        good_false_rejection_all_valid=float((good&~accepted).sum()/n) if n else None,
        accepted_median_deg=float(np.median(errors[accepted])) if accepted.any() else None,
        accepted_p95_deg=float(np.percentile(errors[accepted],95)) if accepted.any() else None)
    for threshold in (5,10,30):
        large=valid&(errors>threshold)
        result[f'over_{threshold}_accepted_fraction']=float((large&accepted).sum()/accepted.sum()) if accepted.any() else None
        result[f'over_{threshold}_passed_all_valid']=float((large&accepted).sum()/n) if n else None
        result[f'over_{threshold}_original_all_valid']=float(large.sum()/n) if n else None
    runs=[];start=None;end=None
    for row,ok in zip(rows,accepted):
        if not ok:
            if start is None:start=row['frame_start_reception_time_s']
            end=row['frame_end_reception_time_s']
        elif start is not None:runs.append(end-start);start=None
    if start is not None:runs.append(end-start)
    result.update(missing_series_count=len(runs),missing_series_max_s=max(runs,default=0),missing_series_total_s=sum(runs),missing_series_durations_s_json=json.dumps(runs))
    result['reasons_json']=json.dumps(dict(Counter(('frontend:'+r['invalid_reason'] if not r['valid'] else decision(features[r['event_id']],config)[1]) for r,ok in zip(rows,accepted) if not ok)),sort_keys=True)
    return result


def mean_or_none(values):
    finite=[float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(finite)) if finite else None


def select_thresholds(development,grid):
    candidates=[];thresholds={}
    for method in METHODS:
        groups=[c for c in development if c[1]==method]
        feasible=[]
        for p in grid['participation']:
            for margin in grid['margin']:
                config=GateConfig(True,p,margin)
                summaries=[(s,stats(r,f,config)) for s,_,_,r,f in groups]
                retention=np.mean([x['retention'] for _,x in summaries])
                loss={cls:mean_or_none([x['good_false_rejection_conditional'] for s,x in summaries if s['source_class']==cls and x['good_count']]) for cls in ('broadband','nonstationary_harmonic')}
                ok=bool(retention>=.7 and all(v is not None and v<=.05 for v in loss.values()))
                passed=np.mean([x['over_10_passed_all_valid'] for _,x in summaries])
                row=dict(method=method,minimum_participation=p,minimum_margin=margin,retention=float(retention),over_10_passed_all_valid=float(passed),feasible=ok,good_false_rejection_by_class_json=json.dumps(loss,sort_keys=True))
                candidates.append(row)
                if ok:feasible.append(row)
        best=min(feasible,key=lambda x:(x['over_10_passed_all_valid'],-x['retention'],x['minimum_participation'],x['minimum_margin'])) if feasible else dict(minimum_participation=0,minimum_margin=0)
        thresholds[method]={k:best[k] for k in ('minimum_participation','minimum_margin')}
    return thresholds,candidates


def freeze(output=OUT):
    m=load(output)
    if (output/'gate_manifest.json').exists():raise FileExistsError('gate already frozen')
    if any((run_dir(output,x['index'])/'checkpoint.json').exists() for x in m['specs'] if x['split']=='evaluation'):
        raise ValueError('evaluation accessed before gate freeze')
    dev=cells(output,'development')
    if len(dev)!=8*2*3:raise ValueError('all eight development streams required before freeze')
    thresholds,candidates=select_thresholds(dev,m['threshold_grid'])
    _write_csv_gz(output/'development_tradeoff.csv.gz',candidates)
    files={p.relative_to(output).as_posix():sha256(p) for i in range(8) for p in run_dir(output,i).iterdir() if p.is_file()}
    files['development_tradeoff.csv.gz']=sha256(output/'development_tradeoff.csv.gz')
    g=dict(schema_version=1,status='frozen_before_evaluation',source_sha256=m['source_sha256'],
        processing_code_sha256=m['processing_code_sha256'],study_manifest_sha256=sha256(output/'study_manifest.json'),
        thresholds=thresholds,development_selection_failed={method:not any(c['feasible'] for c in candidates if c['method']==method) for method in METHODS},development_sha256=files,selection_rule='frozen protocol v1',scores_are_probabilities=False)
    _write_json(output/'gate_manifest.json',g)
    print(json.dumps(thresholds,indent=2))


def availability(rows,features,config,window=3):
    ids=sorted({r['frame_index'] for r in rows})
    frame_stations={i:set() for i in ids};events=[]
    for r in rows:
        if r['valid'] and decision(features[r['event_id']],config)[0]:
            frame_stations[r['frame_index']].add(r['station_id'])
            events.append((r['available_timestamp_s'],r['station_id']))
    # Each publication epoch only sees events already available; delay is not future access.
    epochs=sorted({r['available_timestamp_s'] for r in rows})
    events.sort(); active=deque(); station_count=Counter(); counts=[]; cursor=0
    for epoch in epochs:
        while cursor<len(events) and events[cursor][0]<=epoch:
            t,station=events[cursor];active.append((t,station));station_count[station]+=1;cursor+=1
        while active and active[0][0]<epoch-window:
            _,station=active.popleft();station_count[station]-=1
        counts.append(sum(n>0 for n in station_count.values()))
    result={}
    for n in (2,3):
        result[f'instantaneous_{n}_stations_fraction']=float(np.mean([len(frame_stations[i])>=n for i in ids]))
        result[f'trailing_3s_{n}_stations_fraction']=float(np.mean([c>=n for c in counts]))
    result.update(frame_groups=len(ids),causal_availability_epochs=len(epochs),confirmation_window_s=window)
    return result


def verify(output=OUT):
    m=load(output);g=gate_manifest(output,m)
    count=0
    for s in m['specs']:
        d=run_dir(output,s['index'])
        if (d/'completion.json').exists():
            verify_completion(d,{'kind':'audio','index':s['index'],'study_manifest_sha256':sha256(output/'study_manifest.json')});count+=1
    replays=0
    for d in (output/'replays').glob('*'):
        c=json.loads((d/'completion.json').read_text());verify_completion(d,{k:c[k] for k in ('kind','index','method','gate','gate_manifest_sha256')});replays+=1
    print(json.dumps(dict(verified_audio=count,verified_replays=replays,planned_audio=16,planned_replays=8)))
    return count,replays


def report(output=OUT):
    m=load(output);g=gate_manifest(output,m);measurements=[];windows=[];decisions=[]
    for split in ('development','evaluation'):
        for spec,method,station,rows,features in cells(output,split):
            for mode in ('off','on'):
                config=GateConfig(mode=='on',**g['thresholds'][method])
                for subset in ('all_frames','tracker_stride'):
                    selected=rows if subset=='all_frames' else [r for r in rows if not r['frame_index']%m['processing']['tracker']['frame_stride']]
                    measurements.append(dict(index=spec['index'],split=split,source_class=spec['source_class'],distance_m=spec['distance_m'],replicate=spec['replicate'],method=method,station=station,gate=mode,subset=subset,**stats(selected,features,config)))
                for r in rows:
                    ok,reason=decision(features[r['event_id']],config)
                    decisions.append(dict(index=spec['index'],event_id=r['event_id'],method=method,station=station,frame_index=r['frame_index'],gate=mode,raw_valid=r['valid'],accepted=r['valid'] and ok,reason=('frontend:'+r['invalid_reason'] if not r['valid'] else reason)))
        for spec in m['specs']:
            if spec['split']!=split or not (run_dir(output,spec['index'])/'completion.json').exists():continue
            rows=_measurement_rows(run_dir(output,spec['index'])/'bearing_records.csv.gz');features=read_features(run_dir(output,spec['index']))
            for method in METHODS:
                for mode in ('off','on'):
                    for subset in ('all_frames','tracker_stride'):
                        selected=[r for r in rows if r['estimator_variant']==method and (subset=='all_frames' or not r['frame_index']%32)]
                        windows.append(dict(index=spec['index'],split=split,method=method,gate=mode,subset=subset,**availability(selected,features,GateConfig(mode=='on',**g['thresholds'][method]))))
    _write_csv_gz(output/'bearing_metrics.csv.gz',measurements)
    _write_csv_gz(output/'station_availability.csv.gz',windows)
    _write_csv_gz(output/'gate_decisions.csv.gz',decisions)
    tracks=[json.loads(p.read_text()) for p in sorted((output/'replays').glob('*/summary.json'))]
    if tracks:_write_csv_gz(output/'tracking_metrics.csv.gz',tracks)
    pooled=[]
    for method in METHODS:
        for mode in ('off','on'):
            selected=[r for r in measurements if r['split']=='evaluation' and r['method']==method and r['gate']==mode and r['subset']=='all_frames']
            if not selected:continue
            pooled.append(dict(method=method,gate=mode,retention=float(np.mean([r['retention'] for r in selected])),over_10_passed_all_valid=float(np.mean([r['over_10_passed_all_valid'] for r in selected])),good_false_rejection_conditional=mean_or_none([r['good_false_rejection_conditional'] for r in selected if r['good_count']])))
    ledger=json.loads((output/'execution_ledger.json').read_text())
    _write_json(output/'aggregate.json',dict(evaluation_macro=pooled,thresholds=g['thresholds'],planned_audio=16,planned_replays=8,
        actual_audio_attempts=sum(a['kind']=='audio' for a in ledger['attempts']),actual_audio_complete=sum(a['kind']=='audio' and a['status']=='complete' for a in ledger['attempts']),
        actual_replay_attempts=sum(a['kind']=='replay' for a in ledger['attempts']),actual_replay_complete=sum(a['kind']=='replay' and a['status']=='complete' for a in ledger['attempts']),
        worker_wall_s=ledger['worker_wall_s'],technical_stop=ledger['stopped']))
    print(json.dumps(pooled,indent=2))
