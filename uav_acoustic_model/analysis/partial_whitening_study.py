"""Frozen bounded S8 normalized partial-whitening study; no automatic retry."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from analysis.partial_whitening_frontend import BETAS,METHODS,SRP,estimate_frame,frame_diagnostics
from analysis.bearing_ambiguity_study import source_bank,scenario,recording_key,run_dir,complete
from analysis.localization_error_attribution import (BEARING_COLUMNS,_serialize_bearing,_write_csv_gz,_write_json,_read_csv_gz,_measurement_rows,_make_measurements,_standard_noise_sha256)
from analysis.robust_track_confirmation import _run_tracker
from analysis.study_execution import bounded_process,verify_variant,EMPTY_COLUMNS
from analysis.tracking_uncertainty_quality import count_batch_fits
from analysis.unseen_manoeuvres_and_sources import _frozen_protocol,recording_inputs
from estimators.retarded_ekf_manoeuvre import ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_residual
from simulation.continuous_stream import extract_overlapping_frames
from simulation.multistation_audio import synthesize_multistation_audio
from validation.gazebo_experiment import canonical_sha256,code_sha256,sha256,calibration_from_experiment
from validation.three_station_audio_tracking_study import _unit_direction_at_reception,_scalar_quality
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/partial_whitening_001'
PROTOCOL=ROOT/'PARTIAL_WHITENING_PROTOCOL.md'
MANIFEST=ROOT/'PARTIAL_WHITENING_PROTOCOL_MANIFEST.json'
COLUMNS=(*BEARING_COLUMNS,'frontend_beta','audio_stream_id')
SOURCES=('analysis/partial_whitening_frontend.py','analysis/partial_whitening_study.py','analysis/partial_whitening_report.py',
    'analysis/bearing_ambiguity_study.py','analysis/localization_error_attribution.py',
    'analysis/robust_track_confirmation.py','analysis/study_execution.py',
    'analysis/tracking_uncertainty_quality.py','analysis/unseen_manoeuvres_and_sources.py')


def source_hashes(): return {p:sha256(ROOT/p) for p in SOURCES}


def protocol():
    p=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if sha256(PROTOCOL)!=p['protocol_sha256'] or code_sha256()!=p['processing_code_sha256']:
        raise ValueError('frozen protocol/core code SHA mismatch')
    if sha256(ROOT/'BEARING_AMBIGUITY_PROTOCOL_MANIFEST.json')!=p['previous_protocol_sha256']:
        raise ValueError('historical settings manifest SHA mismatch')
    _frozen_protocol()
    return p


def load(output=OUT):
    p=protocol();m=json.loads((output/'study_manifest.json').read_text(encoding='utf-8'))
    if m['protocol_manifest_sha256']!=sha256(MANIFEST) or m['source_sha256']!=source_hashes():
        raise ValueError('frozen study code/protocol mismatch')
    for key,value in p.items():
        if m.get(key)!=value: raise ValueError('frozen setting changed: '+key)
    if m['recordings']!=recording_inputs(_frozen_protocol()):raise ValueError('recording SHA changed')
    for meta in (output/'source_banks').glob('*.json'):
        b=json.loads(meta.read_text());
        if sha256(meta.with_suffix('.npy'))!=b['file_sha256']:raise ValueError('source bank SHA changed')
    return m


def initialize(output=OUT):
    p=protocol()
    if output.exists():raise FileExistsError('new directory required; accepted results cannot be overwritten')
    output.mkdir(parents=True)
    m={**p,'recordings':recording_inputs(_frozen_protocol()),'source_sha256':source_hashes(),
       'protocol_manifest_sha256':sha256(MANIFEST),'code_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    m['planned_schedule']={}
    for spec in m['specs']:
        info=m['recordings'][recording_key(spec)]
        samples=round(info['duration_s']*m['processing']['audio']['sampling_rate_hz'])
        front=m['processing']['frontend'];frames=1+(samples-front['frame_length'])//front['hop_length']
        m['planned_schedule'][str(spec['index'])]=dict(audio_samples=samples,frames_per_station=frames,stations=3,baseline_rows=frames*6,development_rows=frames*18 if spec['split']=='development' else None)
    _write_json(output/'study_manifest.json',m)
    _write_json(output/'execution_ledger.json',dict(attempts=[],worker_wall_s=0,stopped=False))
    return m


def selection_manifest(output,m):
    g=json.loads((output/'selection_manifest.json').read_text(encoding='utf-8'))
    if g['source_sha256']!=m['source_sha256'] or g['study_manifest_sha256']!=sha256(output/'study_manifest.json'):
        raise ValueError('selection freeze identity mismatch')
    for name,digest in g['development_sha256'].items():
        if sha256(output/name)!=digest:raise ValueError('development SHA changed')
    return g


def evaluation_access(output,m):
    g=selection_manifest(output,m)
    # Require a distinct Git commit of the frozen manifest before any eval access.
    relative=(output/'selection_manifest.json').resolve().relative_to(ROOT.parent).as_posix()
    tracked=subprocess.run(['git','show','HEAD:'+relative],cwd=ROOT,capture_output=True)
    if tracked.returncode or hashlib.sha256(tracked.stdout).hexdigest()!=sha256(output/'selection_manifest.json'):
        raise ValueError('commit frozen selection manifest before evaluation')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['code_git_sha']:
        raise ValueError('separate selection commit required before evaluation')
    return g


def beta_rows(directory,beta,method=None):
    rows=_measurement_rows(directory/'bearing_records.csv.gz')
    return [r for r in rows if float(r['frontend_beta'])==beta and (method is None or r['estimator_variant']==method)]


def variants(spec,selection=None):
    if spec['split']=='development':return {b:METHODS for b in BETAS}
    return {b:tuple(m for m in METHODS if b==1 or selection['selected_beta'][m]==b) for b in BETAS if b==1 or b in selection['selected_beta'].values()}


def verify_completion(directory,context):
    c=json.loads((directory/'completion.json').read_text())
    if c['status']!='complete_verified' or any(c.get(k)!=v for k,v in context.items()):raise ValueError('completion identity mismatch')
    for name,digest in c['files_sha256'].items():
        if sha256(directory/name)!=digest:raise ValueError('completion SHA mismatch: '+name)
    if context['kind']=='audio':
        cp=json.loads((directory/'checkpoint.json').read_text())
        rows=_read_csv_gz(directory/'bearing_records.csv.gz')
        if cp['status']!='complete' or len(rows)!=cp['expected_bearing_rows'] or len({r['event_id'] for r in rows})!=len(rows):
            raise ValueError('incomplete/duplicate bearing schedule')
        expected={(str(b),m,s,str(i)) for b,methods in cp['variants'].items() for m in methods for s in ('S0','S1','S2') for i in range(cp['frames_per_station'])}
        actual={(str(float(r['frontend_beta'])),r['estimator_variant'],r['station_id'],r['frame_index']) for r in rows}
        if expected!=actual:raise ValueError('bearing schedule mismatch')
    return c


def audio_context(output,index):return dict(kind='audio',index=index,study_manifest_sha256=sha256(output/'study_manifest.json'))


def replay_dir(output,index,method,mode):return output/'replays'/f'{index:02d}_{method}_{mode}'


def replay_context(output,index,method,mode):return dict(kind='replay',index=index,method=method,mode=mode,selection_manifest_sha256=sha256(output/'selection_manifest.json'))


def audio_worker(output,index):
    m=load(output);spec=m['specs'][index]
    selected=evaluation_access(output,m) if spec['split']=='evaluation' else None
    enabled=variants(spec,selected)
    directory=run_dir(output,index);directory.mkdir(parents=True,exist_ok=False)
    source,b=source_bank(output,m,spec);stations,trajectory,offset,info=scenario(m,spec)
    audio,front=m['processing']['audio'],m['processing']['frontend']
    stream_id='pw-'+canonical_sha256(dict(protocol=m['protocol_sha256'],recording=info['state_sha256'],spec=spec,source=b['signal_sha256'],processing=m['processing'],code=m['source_sha256']))[:24]
    _write_json(directory/'checkpoint.json',dict(status='synthesizing',audio_stream_id=stream_id,spec=spec))
    stream=synthesize_multistation_audio(stations,trajectory,duration_s=info['duration_s'],reception_start_time_s=0,
        sampling_rate_hz=audio['sampling_rate_hz'],sound_speed=audio['sound_speed_mps'],signal_model='random_broadband',snr_db=10,
        seed=m['seed_root'],chunk_size_samples=audio['chunk_size_samples'],fir_length=audio['fir_length'],geometric_attenuation=True,
        maximum_emitted_frequency_hz=audio['source_maximum_frequency_hz'],external_source_signal=source,
        external_source_start_time_s=b['start_time_s']-info['time_origin_gazebo_s'],noise_model='fixed_reference_snr',reference_distance_m=100,
        source_seed=spec['source_seed'],noise_seed=spec['noise_seed'],reference_source_rms=1)
    serialized=[];stationmap={s.station_id:s for s in stations}
    for ss in stream.stations:
        station=stationmap[ss.station_id]
        if ss.noise_seed!=spec['station_noise_seeds'][ss.station_id]:raise ValueError('actual station noise seed mismatch')
        frames=extract_overlapping_frames(ss.channels,stream.reception_times_s,frame_length=front['frame_length'],hop_length=front['hop_length'])
        if 'planned_schedule' in m and frames.frame_count!=m['planned_schedule'][str(index)]['frames_per_station']:
            raise ValueError('actual frames differ from frozen planned schedule')
        expected=frames.frame_count*len(stations)*sum(len(methods) for methods in enabled.values())
        for fi,frame in enumerate(frames.frames):
            center=float(frames.center_reception_times_s[fi]);end=float(frames.end_reception_times_s[fi])
            emission,truthworld,truthlocal=_unit_direction_at_reception(station,trajectory,center)
            for beta,methods in enabled.items():
                raw=estimate_frame(frame,station.microphone_positions_local_m,stream.sampling_rate_hz,beta=beta,methods=methods)
                if fi in {0,32,frames.frame_count-1}:
                    curves=frame_diagnostics(frame,station.microphone_positions_local_m,stream.sampling_rate_hz,beta,raw[SRP]['direction'] if SRP in raw else np.full(3,np.nan))
                    np.savez_compressed(directory/f'curves_{station.station_id}_{fi}_b{beta}.npz',**curves)
                for method in methods:
                    result=raw[method];valid=bool(result['valid']);local=np.asarray(result['direction'])
                    world=station.local_to_world_direction(local) if valid else np.full(3,np.nan)
                    residual=tangent_residual(truthlocal,local) if valid else np.full(2,np.nan)
                    delivery=front['station_delivery_delay_s'][station.station_id]
                    sequence_id=f'{stream_id}-b{beta}'
                    row=dict(sequence_id=sequence_id,station_id=station.station_id,estimator_variant=method,frame_index=fi,
                        frame_start_reception_time_s=float(frames.start_reception_times_s[fi]),frame_center_reception_time_s=center,
                        frame_end_reception_time_s=end,true_emission_time_s_evaluator_only=emission,
                        available_timestamp_s=end+front['modeled_processing_delay_s']+delivery,
                        modeled_processing_delay_s=front['modeled_processing_delay_s'],modeled_delivery_delay_s=delivery,
                        measured_algorithm_runtime_s=result['total_runtime_s'],effective_station_snr_db=ss.effective_snr_db,
                        valid=valid,invalid_reason='' if valid else 'audio_bearing_invalid',boundary_hit=bool(result['boundary']),
                        truth_local=truthlocal,truth_world=truthworld,estimate_local=local,estimate_world=world,residual_rad=residual,
                        geodesic_error_deg=float(np.rad2deg(np.linalg.norm(residual))),quality_metadata=_scalar_quality(result),
                        source_seed=stream.source_seed,noise_seed=ss.noise_seed,truth_used_by_audio_estimator=False,frame_from_continuous_stream=True)
                    saved=_serialize_bearing(row,sequence_id);saved.update(frontend_beta=beta,audio_stream_id=stream_id);serialized.append(saved)
            if fi%128==0 or fi==frames.frame_count-1:
                _write_csv_gz(directory/'bearing_records.csv.gz',serialized,COLUMNS)
                _write_json(directory/'checkpoint.json',dict(status='extracting',last_frame=fi,station_id=ss.station_id,expected_bearing_rows=expected,frames_per_station=frames.frame_count,variants=enabled))
    _write_json(directory/'audio_manifest.json',dict(audio_stream_id=stream_id,spec=spec,source_bank=b,source_signal_sha256=b['signal_sha256'],
        standardized_noise_sha256=_standard_noise_sha256(stream),station_snr_db={s.station_id:s.effective_snr_db for s in stream.stations},
        recording_sha256=info['state_sha256'],translation_enu_m=offset.tolist(),time_origin_gazebo_s=info['time_origin_gazebo_s'],
        bearing_count=len(serialized),frames_per_station=frames.frame_count,synthesis_count=1,variants=enabled,ambiguity_gate_enabled=False))
    _write_json(directory/'checkpoint.json',dict(status='complete',expected_bearing_rows=expected,actual_bearing_rows=len(serialized),frames_per_station=frames.frame_count,variants=enabled))
    complete(directory,audio_context(output,index))


def replay_worker(output,index,method,mode):
    m=load(output);g=evaluation_access(output,m);spec=m['specs'][index]
    if spec['split']!='evaluation' or spec['distance_m']!=200 or spec['replicate']!=0 or method not in METHODS or mode not in ('baseline','selected'):
        raise ValueError('replay outside frozen matrix')
    source=run_dir(output,index);verify_completion(source,audio_context(output,index))
    beta=1. if mode=='baseline' else g['selected_beta'][method]
    if mode=='selected' and beta==1:raise ValueError('beta=1 selected must reuse verified baseline replay')
    directory=replay_dir(output,index,method,mode);directory.mkdir(parents=True,exist_ok=False)
    t=m['processing']['tracker'];calibration=calibration_from_experiment({'processing':m['processing']})
    measurements=_make_measurements(beta_rows(source,beta,method),calibration,method,'original',t['frame_stride'])
    history=ManoeuvreHistoryConfig(np.asarray(t['qc_m2_s3']),history_step_s=t['history_step_s'],history_window_s=t['history_window_s'],maximum_range_m=t['maximum_range_m'],maximum_transport_delay_s=t['maximum_transport_delay_s'])
    recovery=InitializationRecoveryConfig(**{**t['recovery'],'confirmation_station_count':3})
    stations,trajectory,_,info=scenario(m,spec)
    tracks,updates,diagnostics,summary=_run_tracker(stations,trajectory,measurements,method,'three_station_confirmation',history,recovery,0)
    names={}
    for kind,data in {'tracking':tracks,'updates':updates,**diagnostics}.items():
        name=f'{kind}_{method}_three_station_confirmation.csv.gz';_write_csv_gz(directory/name,data,columns=None if data else EMPTY_COLUMNS[kind]);names[name]=sha256(directory/name)
    evidence=verify_variant(directory,method,'three_station_confirmation',expected_epochs=sorted({x.available_timestamp_s for x in measurements}),allowed_event_ids={bearing_event_id(x) for x in measurements},expected_hashes=names)
    errors=np.asarray([r['position_error_m'] for r in tracks if r['valid']],float)
    summary.update(index=index,source_class=spec['source_class'],method=method,mode=mode,beta=beta,ambiguity_gate_enabled=False,
        baseline_calibrated_R_reused=True,position_over_10m_all_epochs=float(np.count_nonzero(errors>10)/len(tracks)),
        position_over_50m_all_epochs=float(np.count_nonzero(errors>50)/len(tracks)),**count_batch_fits(diagnostics['batch_fits'],int(tracks[-1]['batch_optimization_count'])))
    _write_json(directory/'summary.json',summary);_write_json(directory/'verified_schedule.json',evidence)
    complete(directory,replay_context(output,index,method,mode))


def reuse_baseline(output,index,method):
    """Reuse identical beta=1 without executing another tracker or copying IDs."""
    m=load(output);g=evaluation_access(output,m)
    if g['selected_beta'][method]!=1:raise ValueError('reuse requires beta=1')
    source=replay_dir(output,index,method,'baseline');context=replay_context(output,index,method,'baseline')
    verify_completion(source,context)
    path=output/'replays'/f'{index:02d}_{method}_selected_reuse.json'
    raw=run_dir(output,index)/'bearing_records.csv.gz'
    baseline_ids=[r['event_id'] for r in beta_rows(run_dir(output,index),1.,method)]
    selected_ids=[r['event_id'] for r in beta_rows(run_dir(output,index),g['selected_beta'][method],method)]
    if baseline_ids!=selected_ids:raise ValueError('beta=1 inputs differ')
    payload=dict(kind='identical_beta_one_reuse',beta=1,method=method,index=index,
        input_bearings_sha256=sha256(raw),input_event_ids_sha256=canonical_sha256(baseline_ids),
        baseline_completion_sha256=sha256(source/'completion.json'),baseline_path=source.relative_to(output).as_posix(),
        selection_manifest_sha256=sha256(output/'selection_manifest.json'))
    if path.exists() and json.loads(path.read_text())!=payload:raise ValueError('reuse identity mismatch')
    _write_json(path,payload)


def execute(output,stage):
    m=load(output);g=evaluation_access(output,m) if stage!='development' else None
    tasks=[('audio',s['index'],None,None) for s in m['specs'] if s['split']==stage] if stage in ('development','evaluation') else [
        ('replay',s['index'],method,mode) for s in m['specs'] if s['split']=='evaluation' and s['distance_m']==200 and s['replicate']==0 for method in METHODS for mode in ('baseline','selected')]
    lock=output/'worker.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        for kind,index,method,mode in tasks:
            ledger=json.loads((output/'execution_ledger.json').read_text())
            if ledger['stopped']:raise RuntimeError('technical stop; no automatic retries')
            if kind=='replay' and mode=='selected' and g['selected_beta'][method]==1:
                reuse_baseline(output,index,method);print('verified beta=1 reuse',index,method,flush=True);continue
            key=f'{kind}:{index}:{method}:{mode}'
            directory=run_dir(output,index) if kind=='audio' else replay_dir(output,index,method,mode)
            context=audio_context(output,index) if kind=='audio' else replay_context(output,index,method,mode)
            prior=[a for a in ledger['attempts'] if a['key']==key]
            if prior:
                if prior[-1]['status']!='complete':raise RuntimeError('incomplete attempt; no automatic retry')
                verify_completion(directory,context);print('verified skip',key,flush=True);continue
            limit=m['limits']['audio_timeout_s' if kind=='audio' else 'replay_timeout_s'];counts=Counter(a['kind'] for a in ledger['attempts'])
            if counts[kind]>=m['limits']['audio_streams' if kind=='audio' else 'tracker_replays'] or ledger['worker_wall_s']+limit>m['limits']['total_worker_wall_s']:
                ledger.update(stopped=True,stop_reason='budget_before_start');_write_json(output/'execution_ledger.json',ledger);break
            attempt=dict(key=key,kind=kind,index=index,method=method,mode=mode,status='running',timeout_s=limit)
            ledger['attempts'].append(attempt);_write_json(output/'execution_ledger.json',ledger)
            command=[sys.executable,'-m','analysis.partial_whitening_study',kind+'-worker','--output',str(output),'--index',str(index)]
            if method:command+=['--method',method,'--mode',mode]
            print('starting',key,flush=True);result=bounded_process(command,cwd=ROOT,timeout_s=limit)
            logdir=output/'execution_logs';logdir.mkdir(exist_ok=True);label=f'{kind}_{index}_{method}_{mode}'
            for ext in ('stdout','stderr'):(logdir/f'{label}.{ext}').write_bytes(result[ext].encode())
            attempt.update({k:v for k,v in result.items() if k not in ('stdout','stderr')});ledger['worker_wall_s']+=result['wall_s']
            try:
                if result['timed_out'] or result['returncode']!=0:raise RuntimeError('worker failure/timeout')
                verify_completion(directory,context);attempt['status']='complete';attempt['completion_sha256']=sha256(directory/'completion.json')
            except Exception as exc:attempt['status']='partial';ledger.update(stopped=True,stop_reason=str(exc))
            _write_json(output/'execution_ledger.json',ledger);print(attempt,flush=True)
            if ledger['stopped']:break
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','development','freeze','evaluation','tracking','audio-worker','replay-worker','report','verify'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--index',type=int);parser.add_argument('--method');parser.add_argument('--mode')
    a=parser.parse_args()
    if a.command=='initialize':initialize(a.output)
    elif a.command=='audio-worker':audio_worker(a.output,a.index)
    elif a.command=='replay-worker':replay_worker(a.output,a.index,a.method,a.mode)
    elif a.command in ('development','evaluation','tracking'):execute(a.output,a.command)
    else:
        from analysis.partial_whitening_report import freeze,report,verify
        print({'freeze':freeze,'report':report,'verify':verify}[a.command](a.output))
if __name__=='__main__':main()
