"""At most four protocol-selected, externally supervised ideal-bearing replays."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from analysis.harmonic_tracking_failure import DEFAULT_OUTPUT, PROTOCOL_MANIFEST, verify_sources
from analysis.localization_error_attribution import _measurement_rows, _write_csv_gz, _write_json
from analysis.robust_track_confirmation import _json_rows
from analysis.study_execution import bounded_process, JOURNALS
from analysis.unseen_manoeuvres_and_sources import DEFAULT_OUTPUT as SOURCE, ROOT, _directory, _load, _scenario
from estimators.retarded_ekf_manoeuvre import CausalManoeuvreRetardedTimeEKF, ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from model.measurements import BearingMeasurement
from simulation.moving_source import solve_emission_time
from validation.gazebo_experiment import calibration_from_experiment, sha256, code_sha256

INPUT_VARIANT='ideal_bearing_zero_bias_same_R'


def ideal_inputs(rows, calibrations, stations, trajectory, method, stride, sound_speed):
    station_map={s.station_id:s for s in stations};events=[];audit=[]
    for row in rows:
        if row['estimator_variant']!=method or row['frame_index']%stride:continue
        station=station_map[row['station_id']]
        receive=float(row['frame_center_reception_time_s'])
        emit=float(solve_emission_time(receive,station.position_world_m,trajectory,sound_speed))
        vector=np.asarray(trajectory.q(emit))-station.position_world_m
        direction=station.rotation_local_to_world.T@(vector/np.linalg.norm(vector))
        if not np.allclose(direction,row['truth_local'],atol=1e-10,rtol=1e-10):
            raise ValueError('recomputed retarded direction differs from saved evaluator truth')
        if abs(emit-row['true_emission_time_s_evaluator_only'])>1e-8:
            raise ValueError('recomputed emission time differs from saved evaluator truth')
        common=dict(station_id=row['station_id'],sequence_id=row['sequence_id'],frame_index=row['frame_index'],
                    reception_center_timestamp_s=receive,available_timestamp_s=row['available_timestamp_s'],
                    estimator_variant=method,quality_metadata=row['quality_metadata'])
        calibration=calibrations[row['station_id'],method]
        if row['valid']:
            event=BearingMeasurement(**common,direction_local=direction,
                covariance_tangent_rad2=calibration.covariance_rad2,
                calibration_bias_tangent_rad=np.zeros(2),tangent_frame='prediction')
        else:event=BearingMeasurement.invalid(**common,invalid_reason=row['invalid_reason'])
        if bearing_event_id(event)!=row['event_id']:raise ValueError('event ID changed')
        events.append(event)
        audit.append({'event_id':row['event_id'],'station_id':row['station_id'],'reception_time_s':receive,
                      'availability_time_s':row['available_timestamp_s'],'valid':row['valid'],
                      'emission_time_s_evaluator_only':emit,'ideal_direction_local_json':json.dumps(direction.tolist()),
                      'R_rad2_json':json.dumps(calibration.covariance_rad2.tolist()),'bias_rad_json':'[0.0,0.0]'})
    return tuple(events),audit


def worker(index: int, method: str, directory: Path):
    verify_sources();manifest=_load(SOURCE);spec=manifest['specs'][index]
    frozen=json.loads(PROTOCOL_MANIFEST.read_text())
    if not any(c['index']==index and c['method']==method for c in frozen['replay_cases']):
        raise ValueError('replay outside preselected cases')
    stations,trajectory,_,info=_scenario(manifest,spec)
    settings=manifest['processing']['tracker'];calibrations=calibration_from_experiment({'processing':manifest['processing']})
    rows=_measurement_rows(_directory(SOURCE,spec)/'bearing_records.csv.gz')
    events,audit=ideal_inputs(rows,calibrations,stations,trajectory,method,int(settings['frame_stride']),
                             float(manifest['processing']['audio']['sound_speed_mps']))
    directory.mkdir(parents=True,exist_ok=True)
    _write_csv_gz(directory/'ideal_inputs.csv.gz',audit)
    history=ManoeuvreHistoryConfig(np.asarray(settings['qc_m2_s3']),history_step_s=settings['history_step_s'],
        history_window_s=settings['history_window_s'],maximum_range_m=settings['maximum_range_m'],
        maximum_transport_delay_s=settings['maximum_transport_delay_s'])
    recovery=InitializationRecoveryConfig(**{**settings['recovery'],'confirmation_station_count':3})
    estimator=CausalManoeuvreRetardedTimeEKF(stations,events,estimator_variant=method,
                                           history_config=history,recovery_config=recovery)
    tracks,updates=[],[];epochs=sorted({e.available_timestamp_s for e in events})
    started=time.monotonic()
    # Incremental publications persist even if the parent kills a later optimization.
    with (directory/'publications_checkpoint.jsonl').open('w',encoding='utf-8') as checkpoint:
        for epoch in epochs:
            publication=estimator.advance_to(epoch)
            valid=bool(publication.valid and publication.state is not None)
            row={'processing_time_s':float(epoch),'valid':valid,'confirmed':bool(publication.confirmed),
                 'status':publication.status,'failure_reason':publication.failure_reason or '',
                 'generation':publication.generation,'batch_optimization_count':publication.batch_optimization_count,
                 'truth_used_by_tracker':False,'position_error_m':None,'position_enu_m_json':'','position_covariance_m2_json':''}
            if valid:
                vector=np.asarray(publication.state.vector)
                row.update(position_error_m=float(np.linalg.norm(vector[:3]-trajectory.q(epoch))),
                           position_enu_m_json=json.dumps(vector[:3].tolist()),
                           position_covariance_m2_json=json.dumps(publication.covariance_state[:3,:3].tolist()))
            tracks.append(row);updates.extend(_json_rows(publication.update_diagnostics))
            raw={'publication':row,'batch_fits':_json_rows(estimator.batch_fit_diagnostics),
                 'hypotheses':_json_rows(publication.hypothesis_diagnostics),
                 'lifecycle':_json_rows(publication.lifecycle_diagnostics),
                 'event_uses':_json_rows(publication.event_uses),
                 'candidates':_json_rows(estimator.candidate_diagnostics)}
            checkpoint.write(json.dumps(raw,allow_nan=True)+'\n');checkpoint.flush()
    logs={'tracking':tracks,'updates':updates,'batch_fits':_json_rows(estimator.batch_fit_diagnostics),
          'hypotheses':_json_rows(publication.hypothesis_diagnostics),'lifecycle':_json_rows(publication.lifecycle_diagnostics),
          'event_uses':_json_rows(publication.event_uses),
                 'candidates':_json_rows(estimator.candidate_diagnostics)}
    for kind,data in logs.items():_write_csv_gz(directory/f'{kind}.csv.gz',data,columns=list(data[0]) if data else ('event_id',))
    valid=[t for t in tracks if t['valid']]
    result={'schema_version':1,'status':'complete','index':index,'method':method,'input_variant':INPUT_VARIANT,
        'confirmation_variant':'three_station_confirmation','source_run_id':spec['run_id'],
        'event_count':len(events),'publication_count':len(tracks),'valid_publication_count':len(valid),
        'first_confirmation_s':valid[0]['processing_time_s'] if valid else None,
        'first_position_error_m':valid[0]['position_error_m'] if valid else None,
        'position_rmse_m_conditional':float(np.sqrt(np.mean([t['position_error_m']**2 for t in valid]))) if valid else None,
        'executed_optimizations':sum(f['reason']!='computational_budget_exceeded_before_fit' for f in logs['batch_fits']),
        'worker_wall_s':time.monotonic()-started,'processing_code_sha256':code_sha256(),
        'source_bearing_sha256':sha256(_directory(SOURCE,spec)/'bearing_records.csv.gz'),
        'replay_code_sha256':sha256(Path(__file__)),
        'files_sha256':{p.name:sha256(p) for p in directory.glob('*.csv.gz')}}
    _write_json(directory/'summary.json',result);return result


def run_selected(output=DEFAULT_OUTPUT):
    frozen=verify_sources();root=Path(output)/'ideal_replays';root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();attempts=[]
    ledger=root/'replay_ledger.json'
    for case in frozen['replay_cases']:
        remaining=float(frozen['maximum_total_replay_wall_s'])-(time.monotonic()-started)
        if remaining<=0:break
        directory=root/f"{case['index']:02d}_{case['method']}"
        directory.mkdir()
        attempt={**case,'status':'started','directory':directory.relative_to(Path(output)).as_posix()}
        attempts.append(attempt)
        _write_json(ledger,{'status':'running','attempt_count':len(attempts),'attempts':attempts})
        result=bounded_process([sys.executable,'-m','analysis.harmonic_ideal_replay','worker','--index',str(case['index']),
            '--method',case['method'],'--output',str(directory)],cwd=ROOT,
            timeout_s=min(float(frozen['maximum_wall_s_per_replay']),remaining))
        _write_json(directory/'supervision.json',result)
        attempt.update(status='timed_out' if result['timed_out'] else ('completed' if result['returncode']==0 else 'failed'),
                       wall_s=result['wall_s'])
        if result['returncode']==0:
            summary=json.loads((directory/'summary.json').read_text())
            if summary['status']!='complete':raise ValueError('worker completion missing')
        _write_json(ledger,{'status':'running','attempt_count':len(attempts),'attempts':attempts})
        print(case['index'],case['method'],attempt['status'],round(result['wall_s'],2),flush=True)
    result={'schema_version':1,'status':'finished','attempt_count':len(attempts),
            'maximum_parallel_processes':1,'total_wall_s':time.monotonic()-started,
            'protocol_manifest_sha256':sha256(PROTOCOL_MANIFEST),'attempts':attempts}
    _write_json(ledger,result);verify_sources();return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run-selected','worker'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--index',type=int);p.add_argument('--method')
    a=p.parse_args();result=run_selected(a.output) if a.action=='run-selected' else worker(a.index,a.method,a.output)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
