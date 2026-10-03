"""Read-only audit of the preserved startup failure; never authorizes restart."""
import json
from pathlib import Path
from analysis.bearing_ambiguity_study import OUT,ROOT,MANIFEST,protocol
from analysis.unseen_manoeuvres_and_sources import _frozen_protocol,recording_inputs
from validation.gazebo_experiment import sha256


def audit_stop(output=OUT):
    output=Path(output);p=protocol()
    record=json.loads((output/'technical_failure_manifest.json').read_text())
    if record['schema_version']!=1 or record['status']!='technical_stop_user_declined_restart':
        raise ValueError('unsupported technical failure record')
    for name,digest in record['files_sha256'].items():
        if sha256(output/name)!=digest:raise ValueError(f'failure artifact SHA mismatch: {name}')
    m=json.loads((output/'study_manifest.json').read_text())
    if m['protocol_manifest_sha256']!=sha256(MANIFEST) or m['specs']!=p['specs'] or m['processing']!=p['processing']:
        raise ValueError('startup protocol/matrix/settings changed')
    if m['recordings']!=recording_inputs(_frozen_protocol()):raise ValueError('input recording changed')
    for name,digest in m['source_sha256'].items():
        if sha256(output/record['source_snapshot'][name])!=digest:raise ValueError('startup source snapshot mismatch')
    ledger=json.loads((output/'execution_ledger.json').read_text())
    if not ledger['stopped'] or len(ledger['attempts'])!=1 or ledger['attempts'][0]['status']!='partial':
        raise ValueError('stopped execution ledger changed')
    if list((output/'source_banks').glob('*.npy')) or list((output/'runs').glob('*/checkpoint.json')) or list((output/'replays').glob('*')):
        raise ValueError('unexpected work after the preserved pre-synthesis failure')
    if (output/'gate_manifest.json').exists():raise ValueError('gate must not be frozen without development')
    return dict(status=record['status'],planned_audio_streams=16,planned_replays=8,worker_attempts=1,
        actual_audio_streams=0,actual_replays=0,thresholds_selected=False,evaluation_accessed=False,
        worker_wall_s=ledger['worker_wall_s'],failure='KeyError: spatial (recording key is spatial_manoeuvre)',
        effectiveness='not evaluated; no scientific positive or negative result')


def write_stop_report(output=OUT):
    from analysis.localization_error_attribution import _write_json
    result=audit_stop(output)
    _write_json(Path(output)/'technical_failure_audit.json',result)
    print(json.dumps(result,indent=2))
    return result
