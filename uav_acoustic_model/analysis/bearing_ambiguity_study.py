"""New bounded study; never resumes the historical 15/24 matrix."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from analysis.bearing_ambiguity_gate import (GCC,SRP,GateConfig,ObservableFeatures,apply_gate,decision,frame_features)
from analysis.localization_error_attribution import (BEARING_COLUMNS,_serialize_bearing,_write_csv_gz,_write_json,_read_csv_gz,_measurement_rows,_make_measurements,_standard_noise_sha256)
from analysis.robust_track_confirmation import _run_tracker
from analysis.study_execution import bounded_process,verify_variant,EMPTY_COLUMNS
from analysis.tracking_uncertainty_quality import count_batch_fits
from analysis.unseen_manoeuvres_and_sources import _frozen_protocol,recording_inputs,_bank_support,_scenario
from estimators.retarded_ekf_manoeuvre import ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_residual
from simulation.continuous_stream import extract_overlapping_frames
from simulation.gazebo_offline import load_gazebo_recording
from simulation.multistation_audio import synthesize_multistation_audio
from simulation.signals import random_bandlimited_signal
from validation.gazebo_experiment import canonical_sha256,code_sha256,sha256,stations_from_experiment,calibration_from_experiment
from validation.moving_source_study import estimate_independent_frame
from validation.three_station_audio_tracking_study import _unit_direction_at_reception,_scalar_quality
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/bearing_ambiguity_gate'
PROTOCOL=ROOT/'BEARING_AMBIGUITY_PROTOCOL.md'
MANIFEST=ROOT/'BEARING_AMBIGUITY_PROTOCOL_MANIFEST.json'
SOURCES=(Path(__file__),ROOT/'analysis/bearing_ambiguity_gate.py',ROOT/'analysis/bearing_ambiguity_report.py')
METHODS=(GCC,SRP)


def source_hashes():
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in SOURCES}


def protocol():
    m=json.loads(MANIFEST.read_text())
    if sha256(PROTOCOL)!=m['protocol_sha256'] or code_sha256()!=m['processing_code_sha256']:
        raise ValueError('frozen protocol/processing code SHA mismatch')
    if sha256(ROOT/'UNSEEN_MANOEUVRES_PROTOCOL_MANIFEST.json')!=m['prior_protocol_manifest_sha256']:
        raise ValueError('prior settings SHA mismatch')
    _frozen_protocol()
    return m


def load(output=OUT):
    p=protocol(); m=json.loads((output/'study_manifest.json').read_text())
    if m['protocol_manifest_sha256']!=sha256(MANIFEST) or m['source_sha256']!=source_hashes():
        raise ValueError('study protocol/feature/runner SHA mismatch')
    if m['processing']!=p['processing'] or m['specs']!=p['specs']:
        raise ValueError('frozen processing/matrix changed')
    if m['recordings']!=recording_inputs(_frozen_protocol()):
        raise ValueError('recording SHA changed')
    for meta in (output/'source_banks').glob('*.json'):
        bank=json.loads(meta.read_text())
        if sha256(meta.with_suffix('.npy'))!=bank['file_sha256']:
            raise ValueError('source bank SHA mismatch')
    return m


def initialize(output=OUT, *, authorization_path=None):
    p=protocol()
    if output.exists(): raise FileExistsError('new study directory required')
    authorization=None
    if authorization_path is not None:
        authorization_path=Path(authorization_path)
        authorization=json.loads(authorization_path.read_text())
        if authorization['protocol_manifest_sha256']!=sha256(MANIFEST) or authorization['source_sha256']!=source_hashes():
            raise ValueError('restart authorization protocol/code SHA mismatch')
        if authorization['output_directory']!=output.resolve().relative_to(ROOT).as_posix():
            raise ValueError('restart authorization output mismatch')
        from analysis.bearing_ambiguity_failure import audit_stop
        previous=audit_stop(ROOT/authorization['previous_output_directory'])
        if previous['actual_audio_streams'] or previous['actual_replays']:
            raise ValueError('restart would exceed frozen workload')
        if previous['worker_wall_s']!=authorization['previous_worker_wall_s']:
            raise ValueError('prior worker wall budget changed')
    output.mkdir(parents=True)
    m={**p,'recordings':recording_inputs(_frozen_protocol()),'source_sha256':source_hashes(),
       'protocol_manifest_sha256':sha256(MANIFEST),'code_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    if authorization is not None:
        m['restart_authorization']=authorization
        m['restart_authorization_sha256']=sha256(authorization_path)
    _write_json(output/'study_manifest.json',m)
    _write_json(output/'execution_ledger.json',{'attempts':[],'worker_wall_s':authorization['previous_worker_wall_s'] if authorization else 0,
        'previous_worker_wall_s':authorization['previous_worker_wall_s'] if authorization else 0,'stopped':False})
    return m


def recording_key(spec):
    return {'spatial':'spatial_manoeuvre','radial':'radial_approach_depart'}[spec['trajectory']]


def scenario(manifest,spec):
    return _scenario(manifest,{**spec,'trajectory':recording_key(spec)})


def run_dir(output,index): return output/'runs'/f'{index:02d}'


def harmonic(fs,count,seed,params):
    t=np.arange(count)/fs; a=params
    phase=2*np.pi*(a['f_base']*t+a['f_mod']*(1-np.cos(2*np.pi*a['f_rate']*t))/(2*np.pi*a['f_rate'])+.5*a['chirp']*t*t)
    phases=np.random.default_rng(seed).uniform(0,2*np.pi,a['harmonics'])
    x=(.65+.25*np.sin(2*np.pi*a['env_rate']*t+.4)+.1*np.sin(2*np.pi*.9*t))*sum(k**(-a['power'])*np.sin(k*phase+phases[k-1]) for k in range(1,a['harmonics']+1))
    return x/np.sqrt(np.mean(x*x))


def source_bank(output,m,spec):
    directory=output/'source_banks';directory.mkdir(exist_ok=True)
    path=directory/f"{spec['split']}_{spec['source_class']}_r{spec['replicate']}.npy"
    meta=path.with_suffix('.json')
    if meta.exists():
        b=json.loads(meta.read_text())
        if sha256(path)!=b['file_sha256'] or b['seed']!=spec['source_seed']: raise ValueError('source bank changed')
        return np.load(path,allow_pickle=False),b
    stations=stations_from_experiment({'processing':m['processing']})
    info=m['recordings'][recording_key(spec)]
    recording=load_gazebo_recording(ROOT/info['path'])
    old={**_frozen_protocol(),'geometry':{'initial_distances_m':[200,700]}}
    start,count=_bank_support(recording,stations,info,old,recording_key(spec))
    fs=m['processing']['audio']['sampling_rate_hz']
    if spec['source_class']=='broadband':
        x=random_bandlimited_signal(fs,count,np.random.default_rng(spec['source_seed']),minimum_frequency_hz=300,maximum_frequency_hz=10000,taper_fraction=0)
    else: x=harmonic(fs,count,spec['source_seed'],m['harmonic'][spec['split']])
    np.save(path,x,allow_pickle=False)
    b={'start_time_s':start,'seed':spec['source_seed'],'sample_count':count,'file_sha256':sha256(path),'signal_sha256':hashlib.sha256(x.tobytes()).hexdigest()}
    _write_json(meta,b)
    return x,b


def gate_manifest(output,m):
    path=output/'gate_manifest.json'
    g=json.loads(path.read_text())
    if g['source_sha256']!=m['source_sha256'] or g['study_manifest_sha256']!=sha256(output/'study_manifest.json'):
        raise ValueError('gate freeze identity mismatch')
    for name,digest in g['development_sha256'].items():
        if sha256(output/name)!=digest: raise ValueError('development SHA changed')
    return g


def complete(directory,context):
    files={p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p.name!='completion.json'}
    _write_json(directory/'completion.json',{'status':'complete_verified',**context,'files_sha256':files})


def verify_completion(directory,context):
    c=json.loads((directory/'completion.json').read_text())
    if c['status']!='complete_verified' or any(c.get(k)!=v for k,v in context.items()): raise ValueError('completion context mismatch')
    for name,digest in c['files_sha256'].items():
        if sha256(directory/name)!=digest: raise ValueError(f'completion SHA mismatch: {name}')
    if context.get('kind')=='audio':
        checkpoint=json.loads((directory/'checkpoint.json').read_text())
        if checkpoint['status']!='complete' or checkpoint['actual_bearing_rows']!=checkpoint['expected_bearing_rows']:
            raise ValueError('incomplete bearing schedule')
        rows=_read_csv_gz(directory/'bearing_records.csv.gz');features=_read_csv_gz(directory/'features.csv.gz')
        if len(rows)!=checkpoint['expected_bearing_rows'] or len(features)!=len(rows):
            raise ValueError('incomplete feature/bearing rows')
        if len({r['event_id'] for r in rows})!=len(rows) or {r['event_id'] for r in rows}!={r['event_id'] for r in features}:
            raise ValueError('bearing/feature identity mismatch')
    return c


def audio_worker(output,index):
    m=load(output);spec=m['specs'][index]
    if spec['split']=='evaluation': gate_manifest(output,m)
    directory=run_dir(output,index);directory.mkdir(parents=True,exist_ok=False)
    source,b=source_bank(output,m,spec)
    stations,trajectory,offset,info=scenario(m,spec)
    audio,front=m['processing']['audio'],m['processing']['frontend']
    run_id='amb-'+canonical_sha256({'protocol':m['protocol_sha256'],'recording':info['state_sha256'],'spec':spec,'source_sha':b['signal_sha256'],'processing':m['processing'],'feature_sha':m['source_sha256']})[:24]
    _write_json(directory/'checkpoint.json',{'status':'synthesizing','run_id':run_id,'spec':spec,'expected_audio_samples':round(info['duration_s']*audio['sampling_rate_hz'])})
    stream=synthesize_multistation_audio(stations,trajectory,duration_s=info['duration_s'],reception_start_time_s=0,
        sampling_rate_hz=audio['sampling_rate_hz'],sound_speed=audio['sound_speed_mps'],signal_model='random_broadband',snr_db=10,
        seed=20261003,chunk_size_samples=audio['chunk_size_samples'],fir_length=audio['fir_length'],geometric_attenuation=True,
        maximum_emitted_frequency_hz=audio['source_maximum_frequency_hz'],external_source_signal=source,
        external_source_start_time_s=b['start_time_s']-info['time_origin_gazebo_s'],noise_model='fixed_reference_snr',reference_distance_m=100,
        source_seed=spec['source_seed'],noise_seed=spec['noise_seed'],reference_source_rms=1)
    _write_json(directory/'checkpoint.json',{'status':'extracting','run_id':run_id,'spec':spec,'expected_audio_samples':len(stream.reception_times_s)})
    serialized=[];features=[];stationmap={s.station_id:s for s in stations}
    for ss in stream.stations:
        station=stationmap[ss.station_id]
        frames=extract_overlapping_frames(ss.channels,stream.reception_times_s,frame_length=front['frame_length'],hop_length=front['hop_length'])
        for fi,frame in enumerate(frames.frames):
            raw=estimate_independent_frame(frame,station.microphone_positions_local_m,stream.sampling_rate_hz)
            observed,curves=frame_features(frame,station.microphone_positions_local_m,stream.sampling_rate_hz,raw[SRP]['direction'])
            if fi in {0,32,frames.frame_count-1}:
                np.savez_compressed(directory/f'curves_{station.station_id}_{fi}.npz',**curves)
            center=float(frames.center_reception_times_s[fi]);end=float(frames.end_reception_times_s[fi])
            emission,truthworld,truthlocal=_unit_direction_at_reception(station,trajectory,center)
            for method in METHODS:
                result=raw[method];valid=bool(result['valid']);local=np.asarray(result['direction'])
                world=station.local_to_world_direction(local) if valid else np.full(3,np.nan)
                residual=tangent_residual(truthlocal,local) if valid else np.full(2,np.nan)
                delivery=front['station_delivery_delay_s'][station.station_id]
                row=dict(sequence_id=run_id,station_id=station.station_id,estimator_variant=method,frame_index=fi,
                    frame_start_reception_time_s=float(frames.start_reception_times_s[fi]),frame_center_reception_time_s=center,
                    frame_end_reception_time_s=end,true_emission_time_s_evaluator_only=emission,
                    available_timestamp_s=end+front['modeled_processing_delay_s']+delivery,
                    modeled_processing_delay_s=front['modeled_processing_delay_s'],modeled_delivery_delay_s=delivery,
                    measured_algorithm_runtime_s=result['total_runtime_s'],effective_station_snr_db=ss.effective_snr_db,
                    valid=valid,invalid_reason='' if valid else 'audio_bearing_invalid',boundary_hit=bool(result['boundary']),
                    truth_local=truthlocal,truth_world=truthworld,estimate_local=local,estimate_world=world,residual_rad=residual,
                    geodesic_error_deg=float(np.rad2deg(np.linalg.norm(residual))),quality_metadata=_scalar_quality(result),
                    source_seed=stream.source_seed,noise_seed=ss.noise_seed,truth_used_by_audio_estimator=False,frame_from_continuous_stream=True)
                saved=_serialize_bearing(row,run_id);serialized.append(saved)
                features.append(dict(event_id=saved['event_id'],station_id=station.station_id,estimator_variant=method,frame_index=fi,**asdict(observed[method])))
            if fi%128==0 or fi==frames.frame_count-1:
                # Full partial journals with explicit station/frame checkpoint survive a killed worker.
                _write_csv_gz(directory/'bearing_records.csv.gz',serialized,BEARING_COLUMNS)
                _write_csv_gz(directory/'features.csv.gz',features)
                _write_json(directory/'checkpoint.json',{'status':'extracting','run_id':run_id,'station_id':station.station_id,'last_frame':fi,'expected_frames_per_station':frames.frame_count,'expected_bearing_rows':frames.frame_count*len(stations)*2})
    _write_json(directory/'audio_manifest.json',{'run_id':run_id,'spec':spec,'source_bank':b,'source_signal_sha256':b['signal_sha256'],
        'standardized_noise_sha256':_standard_noise_sha256(stream),'station_snr_db':{s.station_id:s.effective_snr_db for s in stream.stations},
        'recording_sha256':info['state_sha256'],'translation_enu_m':offset.tolist(),'time_origin_gazebo_s':info['time_origin_gazebo_s'],
        'bearing_count':len(serialized),'frames_per_station':frames.frame_count,'synthesis_count':1})
    _write_json(directory/'checkpoint.json',{'status':'complete','expected_bearing_rows':frames.frame_count*len(stations)*2,'actual_bearing_rows':len(serialized)})
    complete(directory,{'index':index,'kind':'audio','study_manifest_sha256':sha256(output/'study_manifest.json')})


def read_features(directory):
    return {r['event_id']:ObservableFeatures(float(r['participation']) if r['participation'] else None,
        float(r['margin']) if r['margin'] else None,r['undefined_reason']) for r in _read_csv_gz(directory/'features.csv.gz')}


def replay_worker(output,index,method,mode):
    m=load(output);g=gate_manifest(output,m);spec=m['specs'][index]
    if spec['split']!='evaluation' or spec['distance_m']!=200 or spec['replicate']!=0 or method not in METHODS or mode not in ('off','on'):
        raise ValueError('replay outside preselected matrix')
    source=run_dir(output,index);verify_completion(source,{'index':index,'kind':'audio','study_manifest_sha256':sha256(output/'study_manifest.json')})
    directory=output/'replays'/f'{index:02d}_{method}_{mode}';directory.mkdir(parents=True,exist_ok=False)
    rows=_measurement_rows(source/'bearing_records.csv.gz');features=read_features(source)
    t=m['processing']['tracker'];calibration=calibration_from_experiment({'processing':m['processing']})
    raw=_make_measurements(rows,calibration,method,'original',t['frame_stride'])
    config=GateConfig(enabled=mode=='on',**g['thresholds'][method])
    measurements=tuple(apply_gate(x,features[bearing_event_id(x)],config) for x in raw)
    journal=[{'event_id':bearing_event_id(x),'valid':x.valid,'reason':x.invalid_reason or '',
              'raw_valid':r.valid,'gate_rejected':r.valid and not x.valid,'available_timestamp_s':x.available_timestamp_s} for r,x in zip(raw,measurements)]
    _write_csv_gz(directory/'gate_events.csv.gz',journal)
    history=ManoeuvreHistoryConfig(np.asarray(t['qc_m2_s3']),history_step_s=t['history_step_s'],history_window_s=t['history_window_s'],maximum_range_m=t['maximum_range_m'],maximum_transport_delay_s=t['maximum_transport_delay_s'])
    recovery=InitializationRecoveryConfig(**{**t['recovery'],'confirmation_station_count':3})
    stations,trajectory,_,info=scenario(m,spec)
    tracks,updates,diagnostics,summary=_run_tracker(stations,trajectory,measurements,method,'three_station_confirmation',history,recovery,0)
    rejected={r['event_id'] for r in journal if r['gate_rejected']}
    for row in diagnostics['event_uses']:
        if row['event_id'] in rejected: raise ValueError('gate-rejected event used by tracker')
    names={}
    for kind,data in {'tracking':tracks,'updates':updates,**diagnostics}.items():
        name=f'{kind}_{method}_three_station_confirmation.csv.gz'
        _write_csv_gz(directory/name,data,columns=None if data else EMPTY_COLUMNS[kind]);names[name]=sha256(directory/name)
    evidence=verify_variant(directory,method,'three_station_confirmation',expected_epochs=sorted({x.available_timestamp_s for x in raw}),allowed_event_ids={bearing_event_id(x) for x in raw},expected_hashes=names)
    errors=np.asarray([r['position_error_m'] for r in tracks if r['valid']],float)
    summary.update(index=index,source_class=spec['source_class'],method=method,gate=mode,
        gate_rejection_count=len(rejected),position_over_10m_all_epochs=float(np.count_nonzero(errors>10)/len(tracks)),
        position_over_50m_all_epochs=float(np.count_nonzero(errors>50)/len(tracks)),
        **count_batch_fits(diagnostics['batch_fits'],int(tracks[-1]['batch_optimization_count'])))
    _write_json(directory/'summary.json',summary)
    _write_json(directory/'verified_schedule.json',evidence)
    complete(directory,{'kind':'replay','index':index,'method':method,'gate':mode,'gate_manifest_sha256':sha256(output/'gate_manifest.json')})


def execute(output,stage):
    m=load(output)
    if stage=='development': tasks=[('audio',x['index'],None,None) for x in m['specs'] if x['split']=='development']
    elif stage=='evaluation':
        gate_manifest(output,m);tasks=[('audio',x['index'],None,None) for x in m['specs'] if x['split']=='evaluation']
    else:
        gate_manifest(output,m);tasks=[('replay',x['index'],method,mode) for x in m['specs'] if x['split']=='evaluation' and x['distance_m']==200 and x['replicate']==0 for method in METHODS for mode in ('off','on')]
    lock=output/'worker.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        for kind,index,method,mode in tasks:
            ledger=json.loads((output/'execution_ledger.json').read_text())
            if ledger['stopped']: raise RuntimeError('technical stop: no automatic retries')
            key=f'{kind}:{index}:{method}:{mode}'
            prior=[a for a in ledger['attempts'] if a['key']==key]
            directory=run_dir(output,index) if kind=='audio' else output/'replays'/f'{index:02d}_{method}_{mode}'
            context=({'kind':'audio','index':index,'study_manifest_sha256':sha256(output/'study_manifest.json')} if kind=='audio' else
                {'kind':'replay','index':index,'method':method,'gate':mode,'gate_manifest_sha256':sha256(output/'gate_manifest.json')})
            if prior:
                if prior[-1]['status']!='complete': raise RuntimeError('incomplete previous attempt; no automatic retry')
                verify_completion(directory,context);print('verified skip',key,flush=True);continue
            limit=m['limits']['audio_timeout_s' if kind=='audio' else 'replay_timeout_s']
            counts=Counter(a['kind'] for a in ledger['attempts'])
            if counts[kind]>=m['limits']['audio_streams' if kind=='audio' else 'tracker_replays'] or ledger['worker_wall_s']+limit>m['limits']['total_worker_wall_s']:
                ledger['stopped']=True;ledger['stop_reason']='budget_before_start';_write_json(output/'execution_ledger.json',ledger);break
            attempt={'key':key,'kind':kind,'index':index,'method':method,'gate':mode,'status':'running','timeout_s':limit}
            ledger['attempts'].append(attempt);_write_json(output/'execution_ledger.json',ledger)
            command=[sys.executable,'-m','analysis.bearing_ambiguity_study',kind+'-worker','--output',str(output),'--index',str(index)]
            if method:command+=['--method',method,'--gate',mode]
            print('starting',key,flush=True)
            result=bounded_process(command,cwd=ROOT,timeout_s=limit)
            logdir=output/'execution_logs';logdir.mkdir(exist_ok=True)
            label=f'{kind}_{index}_{method}_{mode}'
            for ext in ('stdout','stderr'):(logdir/f'{label}.{ext}').write_bytes(result[ext].encode())
            attempt.update({k:v for k,v in result.items() if k not in ('stdout','stderr')})
            ledger['worker_wall_s']+=result['wall_s']
            try:
                if result['timed_out'] or result['returncode']!=0:raise RuntimeError('worker failure/timeout')
                verify_completion(directory,context);attempt['status']='complete'
            except Exception as exc:
                attempt['status']='partial';ledger['stopped']=True;ledger['stop_reason']=str(exc)
            _write_json(output/'execution_ledger.json',ledger)
            print(attempt,flush=True)
            if ledger['stopped']:break
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','development','freeze','evaluation','tracking','audio-worker','replay-worker','report','verify'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--index',type=int);parser.add_argument('--method');parser.add_argument('--gate');parser.add_argument('--authorization',type=Path)
    a=parser.parse_args();out=a.output
    if a.command=='initialize':initialize(out,authorization_path=a.authorization)
    elif a.command=='audio-worker':audio_worker(out,a.index)
    elif a.command=='replay-worker':replay_worker(out,a.index,a.method,a.gate)
    elif a.command in ('development','evaluation','tracking'):execute(out,a.command)
    else:
        if a.command in ('report','verify') and (out/'technical_failure_manifest.json').exists():
            from analysis.bearing_ambiguity_failure import write_stop_report
            write_stop_report(out);return
        from analysis.bearing_ambiguity_report import freeze,report,verify
        {'freeze':freeze,'report':report,'verify':verify}[a.command](out)

if __name__=='__main__':main()
