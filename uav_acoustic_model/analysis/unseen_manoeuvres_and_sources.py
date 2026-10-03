"""Frozen PX4/Gazebo transfer check on unseen manoeuvres and source spectra.

Two physical flight recordings are read only. Each spec synthesizes one shared
continuous audio stream, extracts bearings once, and replays the unchanged
GCC/SRP and baseline/three-station trackers on those saved bearings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from estimators.retarded_ekf_manoeuvre import ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from simulation.gazebo_offline import load_gazebo_recording
from simulation.multistation_audio import _common_source_support, synthesize_multistation_audio
from simulation.signals import random_bandlimited_signal
from validation.gazebo_experiment import canonical_sha256, calibration_from_experiment, code_sha256, sha256, stations_from_experiment
from validation.localization_range_study import _geometry_summary, _translated
from validation.three_station_audio_tracking_study import extract_audio_bearing_records
from analysis.localization_error_attribution import BEARING_COLUMNS, _make_measurements, _measurement_rows, _read_csv_gz, _serialize_bearing, _standard_noise_sha256, _write_csv, _write_csv_gz, _write_json
from analysis.robust_track_confirmation import METHODS, VARIANTS, _run_tracker
from analysis.tracking_uncertainty_quality import count_batch_fits, nominal_coordinate_precision

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'UNSEEN_MANOEUVRES_PROTOCOL.md'
PROTOCOL_MANIFEST = ROOT / 'UNSEEN_MANOEUVRES_PROTOCOL_MANIFEST.json'
RUNNER = Path(__file__).resolve()
FROZEN_RUNNER_V1 = ROOT / 'analysis/frozen/unseen_manoeuvres_and_sources_v1.py'
DEFAULT_OUTPUT = ROOT / 'results' / 'unseen_manoeuvres_and_sources'
SCHEMA_VERSION = 1
BEFORE_FIT = 'computational_budget_exceeded_before_fit'


def _frozen_protocol() -> dict[str, Any]:
    frozen = json.loads(PROTOCOL_MANIFEST.read_text(encoding='utf-8'))
    if frozen['schema_version'] != 1 or frozen['status'] != 'pre_evaluation_frozen':
        raise ValueError('unsupported or unfrozen protocol manifest')
    if sha256(PROTOCOL) != frozen['protocol_sha256']:
        raise ValueError('frozen protocol SHA mismatch')
    prior = ROOT / 'results' / 'robust_track_confirmation' / 'evaluation_manifest.json'
    if sha256(prior) != frozen['prior_evaluation_manifest_sha256']:
        raise ValueError('prior published evaluation changed')
    if code_sha256() != frozen['prior_processing_code_sha256']:
        raise ValueError('frozen processing code changed')
    if canonical_sha256(frozen['processing']) != frozen['prior_processing_sha256']:
        raise ValueError('frozen processing settings changed')
    for item in frozen['flight_plans'].values():
        if sha256(ROOT / item['path']) != item['sha256']:
            raise ValueError('frozen flight plan changed')
    for name, expected in frozen['flight_code_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'frozen flight code changed: {name}')
    if len(frozen['matrix']) != 24 or frozen['tracker_run_count'] != 96:
        raise ValueError('frozen matrix size changed')
    return frozen


def nonstationary_harmonic_signal(sampling_rate_hz: float, count: int, seed: int) -> np.ndarray:
    """Protocol waveform; normalize the complete source bank exactly once."""
    t = np.arange(int(count), dtype=float) / float(sampling_rate_hz)
    fundamental_phase = 2*np.pi*(550*t + 120*(1-np.cos(2*np.pi*.07*t))/(2*np.pi*.07) + 1.25*t*t)
    envelope = .65 + .25*np.sin(2*np.pi*.37*t+.4) + .10*np.sin(2*np.pi*.9*t)
    phases = np.random.default_rng(int(seed)).uniform(0.0, 2*np.pi, 5)
    waveform = envelope * sum(k**-1.1*np.sin(k*fundamental_phase+phases[k-1]) for k in range(1, 6))
    rms = float(np.sqrt(np.mean(waveform**2)))
    if rms <= 0 or not np.isfinite(rms):
        raise ValueError('invalid harmonic source RMS')
    return waveform/rms


def recording_inputs(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    stations = stations_from_experiment({'processing': frozen['processing']})
    for name, item in frozen['flight_plans'].items():
        directory = ROOT / item['recording_directory']
        recording = load_gazebo_recording(directory)
        manifest = recording.manifest
        if manifest['kind'] != 'px4_flight' or sha256(directory/'flight_plan.json') != item['sha256']:
            raise ValueError(f'wrong PX4 flight or plan: {name}')
        if manifest['flight_program']['trajectory_profile'] != name:
            raise ValueError(f'PX4 trajectory profile mismatch: {name}')
        frozen_stations = {station['id']: station for station in frozen['processing']['stations']}
        for station in manifest['station_config']:
            expected = frozen_stations[station['id']]
            if station['position_m'] != expected['position_m'] or station['rpy_rad'] != expected['rpy_rad']:
                raise ValueError(f'station geometry changed in PX4 flight: {name}')
        if manifest['gazebo_seed_applied'] != json.loads((ROOT/item['path']).read_text())['gazebo']['seed']:
            raise ValueError(f'Gazebo seed not applied: {name}')
        phase = {x['name']:x for x in manifest['phase_intervals']}
        start = float(phase['hover_before']['start_s'])-.5
        stop = float(phase['hover_after']['end_s'])+.5
        if start <= manifest['recording_start_s']+.5 or stop >= manifest['recording_end_s']-.5:
            raise ValueError(f'insufficient recorded emission history/tail: {name}')
        if not (start < stop):
            raise ValueError('empty reception window')
        result[name] = {
            'path': item['recording_directory'], 'state_sha256': recording.csv_sha256,
            'manifest_sha256': sha256(directory/'manifest.json'),
            'export_assessment_sha256': sha256(directory/'export_assessment.json'),
            'flight_plan_sha256': item['sha256'], 'reception_start_s': start,
            'duration_s': stop-start, 'export_rate_hz': 1/manifest['export_period_s'],
            'evaluation_phases': [x for x in manifest['phase_intervals'] if x['name'] not in
                                  {'preflight','takeoff','land','landed'}],
        }
        for distance in frozen['geometry']['initial_distances_m']:
            trajectory, _ = _translated(recording, stations, start, distance, name)
            _geometry_summary(trajectory, stations, start, stop-start)
    return result


def _bank_support(recording, stations, info, frozen, trajectory_name) -> tuple[float,int]:
    audio = frozen['processing']['audio']
    fs = float(audio['sampling_rate_hz'])
    reception = float(info['reception_start_s']) + np.arange(round(float(info['duration_s'])*fs), dtype=float)/fs
    starts, ends = [], []
    for distance in frozen['geometry']['initial_distances_m']:
        trajectory, _ = _translated(recording, stations, info['reception_start_s'], distance, trajectory_name)
        start, count = _common_source_support(reception, stations, trajectory, fs,
                                              float(audio['sound_speed_mps']), int(audio['fir_length']))
        starts.append(start); ends.append(start+count/fs)
    bank_start = min(starts)
    return bank_start, int(np.ceil((max(ends)-bank_start)*fs))+1


def _run_identity(frozen: dict, recording_info: dict, bank: dict, spec: dict) -> str:
    return 'unseen-' + canonical_sha256({
        'schema_version': SCHEMA_VERSION, 'protocol_sha256': frozen['protocol_sha256'],
        'processing_sha256': frozen['prior_processing_sha256'],
        'recording_state_sha256': recording_info['state_sha256'],
        'source_signal_sha256': bank['signal_sha256'], 'source_class': spec['source_class'],
        'distance_m': spec['distance_m'], 'replicate': spec['replicate'],
        'source_seed': spec['source_seed'], 'noise_seed': spec['noise_seed'],
        'snr_ref_db': spec['snr_ref_db'],
    })[:24]


def initialize(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'new evaluation directory required: {output}')
    frozen = _frozen_protocol()
    recordings = recording_inputs(frozen)
    stations = stations_from_experiment({'processing':frozen['processing']})
    fs = float(frozen['processing']['audio']['sampling_rate_hz'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'source_banks').mkdir()
    banks = {}
    for trajectory_name in frozen['flight_plans']:
        ti = next(x['trajectory_index'] for x in frozen['matrix'] if x['trajectory']==trajectory_name)
        recording = load_gazebo_recording(ROOT/recordings[trajectory_name]['path'])
        start, count = _bank_support(recording, stations, recordings[trajectory_name], frozen, trajectory_name)
        for source_name in ('broadband','nonstationary_harmonic'):
            seed = next(x['source_seed'] for x in frozen['matrix'] if x['trajectory']==trajectory_name and x['source_class']==source_name)
            if source_name=='broadband':
                source = random_bandlimited_signal(fs,count,np.random.default_rng(seed),
                    minimum_frequency_hz=300.0, maximum_frequency_hz=10000.0,taper_fraction=0.0)
            else:
                source = nonstationary_harmonic_signal(fs,count,seed)
            relative = Path('source_banks')/f'{trajectory_name}_{source_name}.npy'
            np.save(output/relative,source,allow_pickle=False)
            banks[f'{trajectory_name}:{source_name}'] = {
                'path':relative.as_posix(),'start_time_s':start,'sample_count':count,
                'sampling_rate_hz':fs,'seed':seed,'source_class':source_name,
                'rms':float(np.sqrt(np.mean(source**2))),
                'signal_sha256':hashlib.sha256(source.tobytes()).hexdigest(),
                'file_sha256':sha256(output/relative),
            }
    specs=[]
    for item in frozen['matrix']:
        spec=dict(item)
        spec['run_id']=_run_identity(frozen,recordings[item['trajectory']],
                                     banks[f"{item['trajectory']}:{item['source_class']}"],item)
        specs.append(spec)
    if len({x['run_id'] for x in specs})!=24:
        raise ValueError('run ID collision')
    manifest={
        'schema_version':SCHEMA_VERSION,'protocol_sha256':frozen['protocol_sha256'],
        'protocol_manifest_sha256':sha256(PROTOCOL_MANIFEST),'runner_sha256':sha256(RUNNER),
        'processing_code_sha256':code_sha256(),
        'processing_sha256':frozen['prior_processing_sha256'],
        'processing':frozen['processing'],'variants':frozen['variants'],
        'recordings':recordings,'source_banks':banks,'specs':specs,
        'geometry':frozen['geometry'],'fixed_background':frozen['fixed_background'],
        'smoke_index':frozen['smoke_index'],'technical_limits':frozen['technical_limits'],
        'audio_stream_count':24,'tracker_run_count':96,
        'software':{'python':platform.python_version(),'numpy':np.__version__,
                    'scipy':scipy.__version__,'platform':platform.platform()},
    }
    _write_json(output/'evaluation_manifest.json',manifest)
    return manifest


def _load(output: Path) -> dict[str,Any]:
    frozen=_frozen_protocol()
    manifest=json.loads((Path(output)/'evaluation_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version']!=SCHEMA_VERSION or manifest['protocol_sha256']!=frozen['protocol_sha256']:
        raise ValueError('evaluation schema/protocol mismatch')
    # Old manifests refer to the exact preserved v1 bytes; new ones to this version.
    runner_source = (FROZEN_RUNNER_V1 if manifest['runner_sha256'] == sha256(FROZEN_RUNNER_V1) else RUNNER)
    for path,expected,label in ((PROTOCOL_MANIFEST,manifest['protocol_manifest_sha256'],'protocol manifest'),
                                (runner_source,manifest['runner_sha256'],'runner')):
        if sha256(path)!=expected:
            raise ValueError(f'{label} SHA mismatch')
    if manifest['processing_code_sha256']!=code_sha256() or manifest['processing_sha256']!=canonical_sha256(manifest['processing']):
        raise ValueError('processing/code SHA mismatch')
    if manifest['processing']!=frozen['processing'] or manifest['variants']!=frozen['variants']:
        raise ValueError('frozen settings changed')
    if manifest['recordings']!=recording_inputs(frozen):
        raise ValueError('Gazebo recording identity/window changed')
    for key,bank in manifest['source_banks'].items():
        path=Path(output)/bank['path']
        if sha256(path)!=bank['file_sha256']:
            raise ValueError(f'source bank SHA changed: {key}')
    expected=[]
    for item in frozen['matrix']:
        spec=dict(item)
        spec['run_id']=_run_identity(frozen,manifest['recordings'][item['trajectory']],
                                     manifest['source_banks'][f"{item['trajectory']}:{item['source_class']}"],item)
        expected.append(spec)
    if manifest['specs']!=expected:
        raise ValueError('evaluation matrix/IDs changed')
    return manifest


def _directory(output: Path,spec:dict) -> Path:
    return Path(output)/'runs'/f"{spec['index']:02d}_{spec['trajectory']}_{spec['source_class']}_d{spec['distance_m']}_r{spec['replicate']}_{spec['run_id']}"


class RebasedTrajectory:
    """Use small local processing times while retaining recorded Gazebo truth."""
    extrapolate = False

    def __init__(self, base, gazebo_origin_s: float):
        self.base = base
        self.gazebo_origin_s = float(gazebo_origin_s)
        self.knot_times_s = np.asarray(base.knot_times_s, float) - self.gazebo_origin_s
        self.maximum_speed_mps = float(base.maximum_speed_mps)
        self.kind = 'rebased_' + str(base.kind)

    def q(self, time_s):
        return self.base.q(np.asarray(time_s, float) + self.gazebo_origin_s)

    def v(self, time_s):
        return self.base.v(np.asarray(time_s, float) + self.gazebo_origin_s)

    def a(self, time_s):
        return self.base.a(np.asarray(time_s, float) + self.gazebo_origin_s)


def _scenario(manifest:dict,spec:dict):
    stations=stations_from_experiment({'processing':manifest['processing']})
    info=manifest['recordings'][spec['trajectory']]
    recording=load_gazebo_recording(ROOT/info['path'])
    if recording.csv_sha256!=info['state_sha256']:
        raise ValueError('Gazebo state SHA changed')
    absolute_start = float(info['reception_start_s'])
    trajectory,offset=_translated(recording,stations,absolute_start,spec['distance_m'],spec['trajectory'])
    local_info = {**info, 'reception_start_s': 0.0, 'time_origin_gazebo_s': absolute_start}
    return stations,RebasedTrajectory(trajectory,absolute_start),offset,local_info


def _restore_bearings(output:Path,manifest:dict,spec:dict,directory:Path):
    saved=directory/'bearing_records.csv.gz'; metadata_path=directory/'bearing_manifest.json'
    if saved.exists() and metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        if metadata['run_id']!=spec['run_id'] or metadata['bearing_records_sha256']!=sha256(saved):
            raise ValueError('saved bearing cache mismatch')
        return _measurement_rows(saved),metadata,False
    stations,trajectory,offset,info=_scenario(manifest,spec)
    bank=manifest['source_banks'][f"{spec['trajectory']}:{spec['source_class']}"]
    source=np.load(Path(output)/bank['path'],allow_pickle=False)
    if hashlib.sha256(source.tobytes()).hexdigest()!=bank['signal_sha256']:
        raise ValueError('source signal SHA changed')
    audio,front=manifest['processing']['audio'],manifest['processing']['frontend']
    started=time.perf_counter()
    stream=synthesize_multistation_audio(
        stations,trajectory,duration_s=float(info['duration_s']),
        reception_start_time_s=float(info['reception_start_s']),
        sampling_rate_hz=float(audio['sampling_rate_hz']),sound_speed=float(audio['sound_speed_mps']),
        signal_model='random_broadband',snr_db=10.0,seed=20260930,
        chunk_size_samples=int(audio['chunk_size_samples']),fir_length=int(audio['fir_length']),
        geometric_attenuation=True,maximum_emitted_frequency_hz=float(audio['source_maximum_frequency_hz']),
        external_source_signal=source,external_source_start_time_s=float(bank['start_time_s'])-float(info['time_origin_gazebo_s']),
        noise_model='fixed_reference_snr',reference_distance_m=100.0,
        source_seed=int(spec['source_seed']),noise_seed=int(spec['noise_seed']),reference_source_rms=1.0)
    audio_wall=time.perf_counter()-started
    extracted,frontend_wall=extract_audio_bearing_records(
        stream,stations,trajectory,split='unseen_transfer_evaluation',
        configuration_index=int(spec['index']),sequence_index=int(spec['replicate']),
        frame_length=int(front['frame_length']),hop_length=int(front['hop_length']),
        modeled_processing_delay_s=float(front['modeled_processing_delay_s']),
        station_delivery_delay_s=front['station_delivery_delay_s'],sequence_id=spec['run_id'])
    serialized=[_serialize_bearing(row,spec['run_id']) for row in extracted]
    _write_csv_gz(saved,serialized,BEARING_COLUMNS)
    metadata={'schema_version':SCHEMA_VERSION,'run_id':spec['run_id'],'audio_synthesis_count':1,
              'bearing_record_count':len(serialized),'bearing_records_sha256':sha256(saved),
              'recording_state_sha256':info['state_sha256'],'source_signal_sha256':bank['signal_sha256'],
              'standardized_noise_sha256':_standard_noise_sha256(stream),
              'translation_enu_m':offset.tolist(),'time_origin_gazebo_s':info['time_origin_gazebo_s'],
              'audio_wall_s':audio_wall,
              'frontend_wall_s':frontend_wall,
              'actual_station_snr_db':{x.station_id:x.effective_snr_db for x in stream.stations}}
    _write_json(metadata_path,metadata)
    return _measurement_rows(saved),metadata,True


class PeakRSS:
    """Sample process RSS every 20 ms, including audio/front end or tracker work."""
    def __enter__(self):
        import psutil
        self._process=psutil.Process()
        self.peak_bytes=self._process.memory_info().rss
        self._stop=threading.Event()
        def sample():
            while not self._stop.wait(.02):
                self.peak_bytes=max(self.peak_bytes,self._process.memory_info().rss)
        self._thread=threading.Thread(target=sample,daemon=True)
        self._thread.start()
        return self
    def __exit__(self,*args):
        self._stop.set();self._thread.join()
        self.peak_bytes=max(self.peak_bytes,self._process.memory_info().rss)


def summarize_track(tracks:list[dict],updates:list[dict],fits:list[dict],lifecycle:list[dict],
                    *,reception_start_s:float) -> dict[str,Any]:
    if not tracks or any(float(b['processing_time_s'])<=float(a['processing_time_s']) for a,b in zip(tracks,tracks[1:])):
        raise ValueError('empty/nonmonotone track publication schedule')
    valid=[x for x in tracks if str(x['valid']) in ('True','true') or x['valid'] is True]
    confirmed=[x for x in tracks if str(x['confirmed']) in ('True','true') or x['confirmed'] is True]
    first=next((x for x in confirmed if x in valid),None)
    errors=np.asarray([float(x['position_error_m']) for x in valid],float)
    verrors=np.asarray([float(x['velocity_error_mps']) for x in valid],float)
    quality=[]
    for row in tracks:
        is_valid=row in valid
        p=json.loads(row['position_covariance_m2_json']) if is_valid else None
        q=json.loads(row['position_enu_m_json']) if is_valid else None
        quality.append(nominal_coordinate_precision(q,row['status'],p,5.0).status)
    sufficient=[float(row['position_error_m']) for row,label in zip(tracks,quality) if label=='nominal_precision_within_target']
    fit_summary=count_batch_fits(fits,int(tracks[-1]['batch_optimization_count']))
    accepted=sum(str(x['update_applied']) in ('True','true') or x['update_applied'] is True for x in updates)
    reasons=Counter((x.get('failure_reason') or 'unspecified') for x in updates if not (str(x['update_applied']) in ('True','true') or x['update_applied'] is True))
    actions=Counter(x['action'] for x in lifecycle)
    covered=sum(str(x['position_nominal_95_covered']) in ('True','true') or x['position_nominal_95_covered'] is True for x in valid)
    result={
        'ever_confirmed':first is not None,
        'first_confirmation_time_s_relative':float(first['processing_time_s'])-reception_start_s if first else None,
        'first_confirmation_position_error_m':float(first['position_error_m']) if first else None,
        'first_confirmation_velocity_error_mps':float(first['velocity_error_mps']) if first else None,
        'severe_first_confirmation_over_50m':float(first['position_error_m'])>50 if first else False,
        'publication_count':len(tracks),'valid_publication_count':len(valid),
        'availability_fraction':len(valid)/len(tracks),
        'position_rmse_m_conditional':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        'position_p95_m_conditional':float(np.percentile(errors,95)) if len(errors) else None,
        'velocity_rmse_mps_conditional':float(np.sqrt(np.mean(verrors**2))) if len(verrors) else None,
        'velocity_p95_mps_conditional':float(np.percentile(verrors,95)) if len(verrors) else None,
        'position_nees_median':float(np.median([float(x['position_nees']) for x in valid])) if valid else None,
        'position_nominal_95_coverage':covered/len(valid) if valid else None,
        'position_covered_count':covered,
        'nominal_5m_count':len(sufficient),
        'nominal_5m_fraction_all':len(sufficient)/len(tracks),
        'nominal_5m_error_median_m':float(np.median(sufficient)) if sufficient else None,
        'nominal_5m_false_precision_count':sum(x>5 for x in sufficient),
        'nominal_5m_false_precision_fraction':sum(x>5 for x in sufficient)/len(sufficient) if sufficient else None,
        'accepted_update_count':accepted,'rejected_update_count':len(updates)-accepted,
        'update_rejection_reasons_json':json.dumps(dict(reasons),sort_keys=True),
        'reset_count':int(tracks[-1]['reset_count']),
        'reinitialization_count':actions['reinitialized'],
        'lifecycle_actions_json':json.dumps(dict(actions),sort_keys=True),
        'final_status':tracks[-1]['status'],'final_failure_reason':tracks[-1]['failure_reason'],
        **fit_summary,
    }
    return result


def paired_shared(left:list[dict],right:list[dict]) -> dict[str,Any]:
    l={float(x['processing_time_s']):x for x in left};r={float(x['processing_time_s']):x for x in right}
    if len(l)!=len(left) or len(r)!=len(right) or set(l)!=set(r):
        raise ValueError('paired publication times differ or duplicate')
    def valid(x): return str(x['valid']) in ('True','true') or x['valid'] is True
    common=sorted(t for t in l if valid(l[t]) and valid(r[t]))
    result={'shared_valid_count':len(common),
            'baseline_only_valid_count':sum(valid(l[t]) and not valid(r[t]) for t in l),
            'new_only_valid_count':sum(valid(r[t]) and not valid(l[t]) for t in l)}
    for name,rows in (('baseline',l),('new',r)):
        chosen=[rows[t] for t in common]
        result[f'{name}_shared_coverage']=sum(str(x['position_nominal_95_covered']) in ('True','true') or x['position_nominal_95_covered'] is True for x in chosen)/len(chosen) if chosen else None
        for metric in ('position_error_m','velocity_error_mps','position_nees'):
            result[f'{name}_shared_{metric}_median']=float(np.median([float(x[metric]) for x in chosen])) if chosen else None
    return result


def run_one(output:Path,index:int) -> dict[str,Any]:
    """Use the externally supervised v2 envelope; historical artifacts are read only."""
    from analysis.unseen_execution_v2 import run_one as supervised_run_one
    return supervised_run_one(output,index)


def technical_smoke(output:Path=DEFAULT_OUTPUT) -> dict[str,Any]:
    from analysis.study_execution import assert_not_stopped
    assert_not_stopped(output)
    manifest=_load(output)
    index=int(manifest['smoke_index'])
    result=run_one(output,index)
    spec=manifest['specs'][index]
    directory=(Path(output)/result['summary_directory'] if result.get('summary_directory') else _directory(output,spec))
    summary=json.loads((directory/'summary.json').read_text())
    limit=manifest['technical_limits']
    projected=24*float(summary['stream_wall_s'])
    peak=int(summary['stream_sampled_peak_rss_bytes'])
    pass_cost=(summary['stream_wall_s']<=limit['max_wall_s_per_stream'] and
               projected<=limit['max_extrapolated_matrix_wall_s'] and
               peak<=limit['max_peak_rss_bytes'])
    smoke={'schema_version':1,'smoke_index':index,'run_id':spec['run_id'],
           'accuracy_reviewed_for_tuning':False,'run_result':result,
           'stream_wall_s':summary['stream_wall_s'],'stream_sampled_peak_rss_bytes':peak,
           'projected_matrix_wall_s':projected,'technical_limits':limit,
           'technical_cost_pass':pass_cost,
           'bearing_manifest_sha256':summary['bearing_manifest_sha256']}
    path=Path(output)/'technical_smoke.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=smoke: raise ValueError('technical smoke already frozen differently')
    else:_write_json(path,smoke)
    return smoke


def run_all(output:Path=DEFAULT_OUTPUT) -> dict[str,Any]:
    from analysis.study_execution import assert_not_stopped
    assert_not_stopped(output)
    if (Path(output)/'execution_v2/technical_stop.json').exists():
        raise RuntimeError('execution-v2 technical stop blocks run-all')
    manifest=_load(output)
    smoke_path=Path(output)/'technical_smoke.json'
    if not smoke_path.exists() or not json.loads(smoke_path.read_text())['technical_cost_pass']:
        raise ValueError('passing frozen technical smoke required before mass run')
    counts=Counter()
    for spec in manifest['specs']:
        process=subprocess.run([sys.executable,'-m','analysis.unseen_manoeuvres_and_sources','run-one',
                                '--output',str(output),'--index',str(spec['index'])],
                               cwd=ROOT,capture_output=True,text=True,check=False,
                               timeout=float(manifest['technical_limits']['max_wall_s_per_stream'])+5)
        if process.returncode:
            raise RuntimeError(f'run-one index={spec["index"]} failed: {process.stderr[-4000:]}')
        result=json.loads(process.stdout.strip().splitlines()[-1])
        counts[result['status']]+=1
        counts['audio_synthesized_now']+=int(result['audio_synthesized_now'])
        counts['tracker_replays_now']+=int(result['tracker_replays_now'])
        print(f"[{spec['index']+1}/24] {result['status']} {spec['run_id']}",flush=True)
    return dict(counts)


def aggregate(output:Path=DEFAULT_OUTPUT) -> dict[str,Any]:
    output=Path(output);manifest=_load(output)
    rows=[];paired=[];source_hashes={}
    for spec in manifest['specs']:
        directory=_directory(output,spec)
        experiment=json.loads((directory/'experiment.json').read_text())
        if experiment['status']!='complete' or experiment['run_id']!=spec['run_id']:
            raise ValueError(f'incomplete stream {spec["index"]}')
        for name,digest in experiment['result_sha256'].items():
            if sha256(directory/name)!=digest: raise ValueError(f'result SHA mismatch: {name}')
            source_hashes[f'{directory.relative_to(output).as_posix()}/{name}']=digest
        summary=json.loads((directory/'summary.json').read_text())
        if summary['run_id']!=spec['run_id'] or summary['bearing_manifest_sha256']!=sha256(directory/'bearing_manifest.json'):
            raise ValueError('mixed/stale stream summary')
        for result in summary['method_variants']:
            method=result['estimator_variant'];variant=result['confirmation_variant']
            suffix=f'{method}_{variant}.csv.gz'
            tracks=_read_csv_gz(directory/f'tracking_{suffix}')
            updates=_read_csv_gz(directory/f'updates_{suffix}')
            fits=_read_csv_gz(directory/f'batch_fits_{suffix}')
            lifecycle=_read_csv_gz(directory/f'lifecycle_{suffix}')
            derived=summarize_track(tracks,updates,fits,lifecycle,
                                     reception_start_s=0.0)
            for key,value in derived.items():
                stored=result[key]
                if isinstance(value,float):
                    if not np.isclose(value,stored,rtol=1e-10,atol=1e-9):
                        raise ValueError(f'stale summary: {spec["index"]} {method} {variant} {key}')
                elif value!=stored:
                    raise ValueError(f'stale summary: {spec["index"]} {method} {variant} {key}')
            rows.append(result)
        for method in METHODS:
            left=_read_csv_gz(directory/f'tracking_{method}_baseline.csv.gz')
            right=_read_csv_gz(directory/f'tracking_{method}_three_station_confirmation.csv.gz')
            paired.append({'run_id':spec['run_id'],'index':spec['index'],
                           'trajectory':spec['trajectory'],'source_class':spec['source_class'],
                           'distance_m':spec['distance_m'],'replicate':spec['replicate'],
                           'estimator_variant':method,**paired_shared(left,right)})
    if len(rows)!=96 or len(paired)!=48:raise AssertionError('evaluation result count wrong')
    _write_csv(output/'run_summary.csv',rows,list(rows[0]))
    _write_csv(output/'paired_shared.csv',paired,list(paired[0]))
    groups=[]
    for method in METHODS:
        for variant in VARIANTS:
            for trajectory in ('all','spatial_manoeuvre','radial_approach_depart'):
                for source in ('all','broadband','nonstationary_harmonic'):
                    subset=[r for r in rows if r['estimator_variant']==method and r['confirmation_variant']==variant
                            and (trajectory=='all' or r['trajectory']==trajectory)
                            and (source=='all' or r['source_class']==source)]
                    if not subset:continue
                    confirms=[r for r in subset if r['ever_confirmed']]
                    pubs=sum(r['publication_count'] for r in subset)
                    valid=sum(r['valid_publication_count'] for r in subset)
                    nom=sum(r['nominal_5m_count'] for r in subset)
                    group={'estimator_variant':method,'confirmation_variant':variant,
                           'trajectory':trajectory,'source_class':source,'stream_count':len(subset),
                           'confirmed_stream_count':len(confirms),
                           'severe_first_over_50m_count':sum(r['severe_first_confirmation_over_50m'] for r in subset),
                           'publication_count':pubs,'valid_publication_count':valid,
                           'availability_fraction':valid/pubs,
                           'median_first_confirmation_time_s':float(np.median([r['first_confirmation_time_s_relative'] for r in confirms])) if confirms else None,
                           'median_first_error_m':float(np.median([r['first_confirmation_position_error_m'] for r in confirms])) if confirms else None,
                           'median_conditional_position_rmse_m':float(np.median([r['position_rmse_m_conditional'] for r in subset if r['position_rmse_m_conditional'] is not None])) if valid else None,
                           'median_conditional_position_p95_m':float(np.median([r['position_p95_m_conditional'] for r in subset if r['position_p95_m_conditional'] is not None])) if valid else None,
                           'median_conditional_velocity_rmse_mps':float(np.median([r['velocity_rmse_mps_conditional'] for r in subset if r['velocity_rmse_mps_conditional'] is not None])) if valid else None,
                           'position_coverage_pooled':sum(r['position_covered_count'] for r in subset)/valid if valid else None,
                           'nominal_5m_count':nom,'nominal_5m_fraction_all':nom/pubs,
                           'nominal_5m_false_precision_count':sum(r['nominal_5m_false_precision_count'] for r in subset),
                           'nominal_5m_false_precision_fraction':sum(r['nominal_5m_false_precision_count'] for r in subset)/nom if nom else None,
                           'executed_fit_count':sum(r['total_executed_optimizations'] for r in subset),
                           'attempts_rejected_before_fit':sum(r['attempts_rejected_before_optimization'] for r in subset),
                           'tracker_wall_s_total':sum(r['tracker_wall_s'] for r in subset),
                           'stream_peak_rss_bytes_max':max(json.loads((_directory(output,manifest['specs'][r['index']])/'summary.json').read_text())['stream_sampled_peak_rss_bytes'] for r in subset)}
                    groups.append(group)
    _write_csv(output/'group_summary.csv',groups,list(groups[0]))
    result={'schema_version':SCHEMA_VERSION,'audio_stream_count':24,'tracker_run_count':96,
            'source_manifest_sha256':sha256(output/'evaluation_manifest.json'),
            'source_artifact_sha256':source_hashes,
            'tables_sha256':{name:sha256(output/name) for name in ('run_summary.csv','paired_shared.csv','group_summary.csv')},
            'technical_smoke_sha256':sha256(output/'technical_smoke.json'),
            'no_model_tuning_on_evaluation':True}
    _write_json(output/'evaluation_summary.json',result)
    return {k:v for k,v in result.items() if k!='source_artifact_sha256'}


def verify(output:Path=DEFAULT_OUTPUT) -> dict[str,Any]:
    output=Path(output);manifest=_load(output)
    summary=json.loads((output/'evaluation_summary.json').read_text())
    if summary['source_manifest_sha256']!=sha256(output/'evaluation_manifest.json'):
        raise ValueError('evaluation manifest SHA changed')
    if summary['technical_smoke_sha256']!=sha256(output/'technical_smoke.json'):
        raise ValueError('technical smoke SHA changed')
    for name,digest in summary['tables_sha256'].items():
        if sha256(output/name)!=digest:raise ValueError(f'derived table SHA changed: {name}')
    for name,digest in summary['source_artifact_sha256'].items():
        if sha256(output/name)!=digest:raise ValueError(f'source artifact SHA changed: {name}')
    if len(manifest['specs'])!=24 or summary['tracker_run_count']!=96:
        raise ValueError('matrix count mismatch')
    return {'status':'verified','audio_stream_count':24,'tracker_run_count':96}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('init','smoke','run-one','run-all','aggregate','verify'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.action=='init':result=initialize(args.output)
    elif args.action=='smoke':result=technical_smoke(args.output)
    elif args.action=='run-one':
        if args.index is None:parser.error('--index required for run-one')
        result=run_one(args.output,args.index)
    elif args.action=='run-all':result=run_all(args.output)
    elif args.action=='aggregate':result=aggregate(args.output)
    else:result=verify(args.output)
    if args.action=='init':
        print(json.dumps({'status':'initialized','spec_count':len(result['specs']),
                          'source_bank_count':len(result['source_banks'])},sort_keys=True))
    else:print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
