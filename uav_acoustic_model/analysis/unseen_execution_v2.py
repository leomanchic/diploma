"""Version 2 execution envelope for the immutable unseen study.

Historical identities/artifacts remain read only. Continuation requires an
explicit, separately frozen authorization; this module does not grant it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from analysis.localization_error_attribution import _make_measurements, _measurement_rows, _read_csv_gz, _write_csv_gz, _write_json
from analysis.robust_track_confirmation import METHODS, VARIANTS, _run_tracker
from analysis.study_execution import (EXECUTION_VERSION, JOURNALS, assert_not_stopped, bounded_process,
    completion_context, read_completion, verify_variant, write_completion, EMPTY_COLUMNS)
from estimators.retarded_ekf_manoeuvre import ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from validation.gazebo_experiment import calibration_from_experiment, sha256


def _source():
    from analysis import unseen_manoeuvres_and_sources as study
    return study


def _stop_path(output):
    # A new timeout invalidates an authorization pinned to the older stop.
    latest=Path(output)/'execution_v2/technical_stop.json'
    return latest if latest.exists() else Path(output)/'technical_stop.json'


def check_authorization(output, index, authorization):
    stop=_stop_path(output)
    if authorization is None:
        assert_not_stopped(output)
        if stop.exists():raise RuntimeError('execution_v2 technical stop blocks ordinary execution')
        return
    auth=json.loads(Path(authorization).read_text(encoding='utf-8'))
    if not stop.exists():raise ValueError('continuation authorization requires an existing stop marker')
    required={'schema_version':1,'action':'authorized_continuation','execution_version':2,
              'evaluation_manifest_sha256':sha256(Path(output)/'evaluation_manifest.json'),
              'technical_stop_sha256':sha256(stop)}
    if any(auth.get(k)!=v for k,v in required.items()) or index not in auth.get('allowed_indices',[]):
        raise ValueError('continuation authorization scope/provenance mismatch')
    protocol=Path(authorization).parent/auth['protocol_path']
    if sha256(protocol)!=auth['protocol_sha256']:raise ValueError('continuation protocol SHA mismatch')
    if not auth.get('approved_by_user'):raise ValueError('continuation approval is absent')


def variant_inputs(output, manifest, spec, method):
    study=_source();source=study._directory(Path(output),spec)
    rows=_measurement_rows(source/'bearing_records.csv.gz')
    calibration=calibration_from_experiment({'processing':manifest['processing']})
    events=_make_measurements(rows,calibration,method,'original',int(manifest['processing']['tracker']['frame_stride']))
    return events,sorted({e.available_timestamp_s for e in events})


def verified_historical_variant(output, manifest, spec, method, variant):
    study=_source();directory=study._directory(Path(output),spec)
    experiment_path=directory/'experiment.json'
    if not experiment_path.exists():return None
    experiment=json.loads(experiment_path.read_text(encoding='utf-8'))
    if experiment.get('run_id')!=spec['run_id'] or experiment.get('spec')!=spec:
        raise ValueError('historical run identity/spec mismatch')
    bearing=json.loads((directory/'bearing_manifest.json').read_text(encoding='utf-8'))
    if bearing['run_id']!=spec['run_id'] or sha256(directory/'bearing_records.csv.gz')!=bearing['bearing_records_sha256']:
        raise ValueError('historical bearing integrity mismatch')
    expected=experiment.get('result_sha256',{}) if experiment.get('status')=='complete' else {}
    if expected:
        for name,digest in expected.items():
            if not (directory/name).exists() or sha256(directory/name)!=digest:
                raise ValueError(f'historical artifact SHA mismatch: {name}')
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        if summary['run_id']!=spec['run_id'] or summary['bearing_manifest_sha256']!=sha256(directory/'bearing_manifest.json'):
            raise ValueError('historical summary provenance mismatch')
    if not expected:
        audit_path=Path(output)/'partial_audit.json'
        if not audit_path.exists():return None
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if audit['evaluation_manifest_sha256']!=sha256(Path(output)/'evaluation_manifest.json'):
            raise ValueError('partial import audit refers to another evaluation')
        prefix=directory.relative_to(Path(output)).as_posix()+'/'
        expected={name[len(prefix):]:digest for name,digest in audit['artifact_sha256'].items() if name.startswith(prefix)}
    for name in ('bearing_records.csv.gz','bearing_manifest.json'):
        if name not in expected or sha256(directory/name)!=expected[name]:
            raise ValueError(f'historical cache SHA mismatch: {name}')
    names=[f'{kind}_{method}_{variant}.csv.gz' for kind in JOURNALS]
    if not all(name in expected and (directory/name).exists() for name in names):
        if experiment.get('status')=='complete':raise ValueError('historical complete stream missing full journal set')
        return None
    events,epochs=variant_inputs(output,manifest,spec,method)
    evidence=verify_variant(directory,method,variant,expected_epochs=epochs,
                            allowed_event_ids=[bearing_event_id(e) for e in events],expected_hashes=expected)
    return directory,evidence,events


def run_one(output, index, authorization=None):
    study=_source();output=Path(output)
    check_authorization(output,index,authorization)  # before loading/synthesizing anything
    manifest=study._load(output)
    if not 0<=index<len(manifest['specs']):raise IndexError('outside frozen matrix')
    spec=manifest['specs'][index]
    old=study._directory(output,spec)/'experiment.json'
    if old.exists() and json.loads(old.read_text()).get('status')=='complete':
        for method in METHODS:
            for variant in VARIANTS:
                if verified_historical_variant(output,manifest,spec,method,variant) is None:
                    raise ValueError('complete stream lacks verified variants')
        return {'status':'skipped_verified','index':index,'audio_synthesized_now':False,'tracker_replays_now':0}
    timeout=float(manifest['technical_limits']['max_wall_s_per_stream'])
    control=output/'execution_v2';control.mkdir(exist_ok=True)
    command=[sys.executable,'-m','analysis.unseen_execution_v2','worker','--output',str(output),
             '--index',str(index),'--deadline-monotonic',str(time.monotonic()+timeout)]
    if authorization is not None:command.extend(['--authorization',str(Path(authorization).resolve())])
    _write_json(control/'execution_version.json',{'execution_version':EXECUTION_VERSION,
        'evaluation_manifest_sha256':sha256(output/'evaluation_manifest.json'),
        'historical_runner_sha256':manifest['runner_sha256'],
        'runner_v2_sha256':sha256(Path(__file__)),'envelope_sha256':sha256(Path(__file__).with_name('study_execution.py')),
        'authorization_sha256':sha256(Path(authorization)) if authorization else None})
    result=bounded_process(command,cwd=study.ROOT,timeout_s=timeout)
    _write_json(control/f'supervision_{index:02d}.json',result)
    if result['timed_out'] or result['returncode']==124:
        _write_json(control/'technical_stop.json',{'schema_version':2,'status':'stopped_by_external_timeout',
            'trigger_index':index,'limit_wall_s_per_stream':timeout,
            'evaluation_manifest_sha256':sha256(output/'evaluation_manifest.json'),'wall_s':result['wall_s']})
        raise TimeoutError(f'index {index} stopped by external wall timeout; partial execution-v2 journals preserved')
    if result['returncode']!=0:raise RuntimeError(f'execution-v2 worker failed: {result["stderr"][-3000:]}')
    answer=json.loads(result['stdout'].strip().splitlines()[-1])
    answer['worker_stream_wall_s']=answer['stream_wall_s']
    answer['stream_wall_s']=result['wall_s']
    _write_json(output/answer['summary_directory']/'summary.json',answer)
    return answer


def worker(output, index, deadline_monotonic, authorization=None):
    study=_source();output=Path(output);check_authorization(output,index,authorization)
    remaining=float(deadline_monotonic)-time.monotonic()
    if remaining<=0:raise TimeoutError('worker deadline expired')
    watchdog=threading.Timer(remaining,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    started=time.monotonic()
    manifest=study._load(output);spec=manifest['specs'][index]
    original=study._directory(output,spec)
    directory=output/'execution_v2/runs'/original.name;directory.mkdir(parents=True,exist_ok=True)
    settings=manifest['processing']['tracker']
    stations,trajectory,_,info=study._scenario(manifest,spec)
    history=ManoeuvreHistoryConfig(np.asarray(settings['qc_m2_s3']),history_step_s=settings['history_step_s'],
        history_window_s=settings['history_window_s'],maximum_range_m=settings['maximum_range_m'],
        maximum_transport_delay_s=settings['maximum_transport_delay_s'])
    calibrations=calibration_from_experiment({'processing':manifest['processing']})
    # Restore an immutable accepted cache; any new cache is written only into v2.
    cache=original if (original/'bearing_manifest.json').exists() else directory
    if cache==original:
        bearing=json.loads((cache/'bearing_manifest.json').read_text())
        if bearing['run_id']!=spec['run_id'] or sha256(cache/'bearing_records.csv.gz')!=bearing['bearing_records_sha256']:
            raise ValueError('source cache integrity mismatch')
        rows=_measurement_rows(cache/'bearing_records.csv.gz');synthesized=False
    else:rows,bearing,synthesized=study._restore_bearings(output,manifest,spec,directory)
    summaries=[];replays=0
    with study.PeakRSS() as memory:
        for method in METHODS:
            events=_make_measurements(rows,calibrations,method,'original',int(settings['frame_stride']))
            epochs=sorted({e.available_timestamp_s for e in events});ids=[bearing_event_id(e) for e in events]
            for variant in VARIANTS:
                context=completion_context(run_id=spec['run_id'],method=method,variant=variant,
                    bearing_sha=sha256(cache/'bearing_records.csv.gz'),processing_sha=manifest['processing_sha256'],
                    evaluation_sha=sha256(output/'evaluation_manifest.json'))
                completion=directory/f'completion_{method}_{variant}.json'
                suffix=f'{method}_{variant}.csv.gz'
                if completion.exists():
                    claim=json.loads(completion.read_text());journal_directory=output/claim['journal_directory']
                    read_completion(completion,context,journal_directory,expected_epochs=epochs,allowed_event_ids=ids)
                else:
                    imported=verified_historical_variant(output,manifest,spec,method,variant) if cache==original else None
                    if imported:
                        journal_directory,evidence,_=imported;origin='verified_historical_import'
                    else:
                        journal_directory=directory;origin='execution_v2_worker'
                        recovery=InitializationRecoveryConfig(**{**settings['recovery'],
                            'confirmation_station_count':manifest['variants'][variant]['confirmation_station_count']})
                        tracks,updates,diagnostics,_=_run_tracker(stations,trajectory,events,method,variant,
                                                                 history,recovery,float(info['reception_start_s']))
                        replays+=1
                        for kind,data in {'tracking':tracks,'updates':updates,**diagnostics}.items():
                            _write_csv_gz(directory/f'{kind}_{suffix}',data,
                                          columns=list(data[0]) if data else EMPTY_COLUMNS[kind])
                        expected={f'{k}_{suffix}':sha256(directory/f'{k}_{suffix}') for k in JOURNALS}
                        evidence=verify_variant(directory,method,variant,expected_epochs=epochs,
                                                allowed_event_ids=ids,expected_hashes=expected)
                    write_completion(completion,{**context,'journal_directory':journal_directory.relative_to(output).as_posix()},
                                     evidence,origin=origin)
                derived=study.summarize_track(_read_csv_gz(journal_directory/f'tracking_{suffix}'),
                    _read_csv_gz(journal_directory/f'updates_{suffix}'),
                    _read_csv_gz(journal_directory/f'batch_fits_{suffix}'),
                    _read_csv_gz(journal_directory/f'lifecycle_{suffix}'),reception_start_s=0.)
                summaries.append({'run_id':spec['run_id'],'estimator_variant':method,'confirmation_variant':variant,**derived})
                _write_json(directory/'progress.json',{'status':'partial','completed_variant_count':len(summaries),
                                                       'tracker_replays_now':replays})
    result={'execution_version':2,'status':'completed','index':index,'run_id':spec['run_id'],
            'audio_synthesized_now':synthesized,'tracker_replays_now':replays,
            'summary_directory':directory.relative_to(output).as_posix(),
            'stream_wall_s':time.monotonic()-started,'stream_sampled_peak_rss_bytes':memory.peak_bytes,
            'audio_synthesis_count':bearing['audio_synthesis_count'],
            'bearing_manifest_sha256':sha256(cache/'bearing_manifest.json'),'method_variants':summaries}
    _write_json(directory/'summary.json',result);watchdog.cancel();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('continue-one','worker'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--authorization',type=Path);parser.add_argument('--deadline-monotonic',type=float)
    args=parser.parse_args()
    if args.action=='continue-one':
        if args.authorization is None:parser.error('explicit frozen --authorization required')
        result=run_one(args.output,args.index,args.authorization)
    else:
        if args.deadline_monotonic is None:parser.error('parent-supplied --deadline-monotonic required')
        result=worker(args.output,args.index,args.deadline_monotonic,args.authorization)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
