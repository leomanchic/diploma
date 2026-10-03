"""Versioned execution supervision; no acoustic/tracker algorithm changes."""
from __future__ import annotations

import json
import csv
import gzip
import os
import signal
import subprocess
import time
from pathlib import Path

from analysis.localization_error_attribution import _read_csv_gz, _write_json
from validation.gazebo_experiment import canonical_sha256, sha256

EXECUTION_VERSION = 2
JOURNALS = ('tracking','updates','batch_fits','hypotheses','event_uses','lifecycle')
EMPTY_COLUMNS = {
    'tracking': ('processing_time_s','valid','failure_reason'),
    'updates': ('event_id','update_applied','failure_reason'),
    'batch_fits': ('processing_time_s','reason'),
    'hypotheses': ('processing_time_s','action','reason'),
    'event_uses': ('event_id','role'),
    'lifecycle': ('processing_time_s','action','reason'),
}


def assert_not_stopped(output: Path) -> None:
    if any((Path(output)/name).exists() for name in ('technical_stop.json','execution_v2/technical_stop.json')):
        raise RuntimeError('technical_stop.json is present: ordinary run-one/smoke/run-all are blocked; '
                           'a separate frozen continuation authorization and execution-v2 directory are required')


def bounded_process(command, *, cwd, timeout_s: float) -> dict:
    """Enforce wall time at the parent; reap the worker and retain stdout/stderr."""
    if timeout_s <= 0:
        raise ValueError('positive external timeout required')
    started=time.monotonic()
    kwargs={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP}
    process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,**kwargs)
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        timed_out=True
        if os.name!='nt':
            os.killpg(process.pid,signal.SIGKILL)
        else:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,check=False)
            if process.poll() is None:process.kill()
        stdout,stderr=process.communicate(timeout=10)
    return {'execution_version':EXECUTION_VERSION,'timed_out':timed_out,
            'returncode':process.returncode,'wall_s':time.monotonic()-started,
            'stdout':stdout,'stderr':stderr}


def verify_variant(directory: Path, method: str, variant: str, *, expected_epochs,
                   allowed_event_ids, expected_hashes: dict) -> dict:
    """Completion requires all six readable, claimed journals and the entire schedule."""
    import numpy as np
    directory=Path(directory);allowed=set(allowed_event_ids)
    names=[f'{kind}_{method}_{variant}.csv.gz' for kind in JOURNALS]
    for name in names:
        if name not in expected_hashes or not (directory/name).is_file():
            raise ValueError(f'incomplete variant: missing claimed journal {name}')
        if sha256(directory/name)!=expected_hashes[name]:raise ValueError(f'variant SHA mismatch: {name}')
    for kind,name in zip(JOURNALS,names):
        with gzip.open(directory/name,'rt',newline='',encoding='utf-8') as stream:
            header=csv.DictReader(stream).fieldnames
        if header is None or not set(EMPTY_COLUMNS[kind])<=set(header):
            raise ValueError(f'incomplete variant: invalid journal schema {name}')
    logs={kind:_read_csv_gz(directory/name) for kind,name in zip(JOURNALS,names)}
    actual=[float(r['processing_time_s']) for r in logs['tracking']]
    if not actual or not np.array_equal(np.asarray(actual),np.asarray(expected_epochs,float)):
        raise ValueError('incomplete variant: publication schedule not complete or changed')
    if any(row.get('valid')=='True' for row in logs['tracking']) and not logs['event_uses']:
        raise ValueError('incomplete variant: confirmed publication lacks event-use evidence')
    for kind,rows in logs.items():
        for row in rows:
            if row.get('event_id') and row['event_id'] not in allowed:
                raise ValueError(f'foreign event in {kind}')
            for key in ('event_ids','construction_event_ids','confirmation_event_ids','excluded_event_ids'):
                if row.get(key) and not set(json.loads(row[key]))<=allowed:
                    raise ValueError(f'foreign event IDs in {kind}/{key}')
    return {'publication_count':len(actual),'last_publication_s':actual[-1],
            'publication_schedule_sha256':canonical_sha256(actual),
            'files_sha256':{name:sha256(directory/name) for name in names}}


def completion_context(*, run_id: str, method: str, variant: str, bearing_sha: str,
                       processing_sha: str, evaluation_sha: str) -> dict:
    return {'execution_version':EXECUTION_VERSION,'run_id':run_id,'method':method,
            'confirmation_variant':variant,'bearing_sha256':bearing_sha,
            'processing_sha256':processing_sha,'evaluation_sha256':evaluation_sha}


def write_completion(path: Path, context: dict, evidence: dict, *, origin: str) -> dict:
    result={**context,'status':'complete_verified','origin':origin,**evidence}
    _write_json(path,result)
    return result


def read_completion(path: Path, context: dict, directory: Path, *, expected_epochs, allowed_event_ids) -> dict:
    result=json.loads(Path(path).read_text(encoding='utf-8'))
    if result.get('status')!='complete_verified' or any(result.get(k)!=v for k,v in context.items()):
        raise ValueError('variant completion provenance mismatch')
    evidence=verify_variant(directory,context['method'],context['confirmation_variant'],
                            expected_epochs=expected_epochs,allowed_event_ids=allowed_event_ids,
                            expected_hashes=result['files_sha256'])
    if any(result.get(k)!=v for k,v in evidence.items()):raise ValueError('stale completion evidence')
    return result
