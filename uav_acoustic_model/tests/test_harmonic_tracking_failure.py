"""Diagnostic mathematics and supervisor regressions; no audio/flight runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.harmonic_tracking_failure import (DEFAULT_OUTPUT, calibration_diagnostics,
    large_error_runs, safe_corr, verify, verify_sources)
from analysis.localization_error_attribution import _write_csv_gz, _measurement_rows
from analysis.study_execution import (JOURNALS, EMPTY_COLUMNS, bounded_process, completion_context,
    read_completion, verify_variant, write_completion)
from analysis import unseen_manoeuvres_and_sources as study
from analysis.unseen_execution_v2 import check_authorization, verified_historical_variant
from validation.gazebo_experiment import sha256


def test_frozen_inputs_and_published_study_still_load_without_rewriting_ids():
    manifest=study._load(study.DEFAULT_OUTPUT)
    frozen=verify_sources()
    assert sha256(study.FROZEN_RUNNER_V1)==manifest['runner_sha256']==frozen['frozen_runner_sha256']
    assert sha256(study.RUNNER)!=manifest['runner_sha256']
    assert manifest['specs'][6]['run_id']=='unseen-beffd269c4dc99056f925c98'
    assert verify(DEFAULT_OUTPUT)['historical_inputs_unchanged']


@pytest.mark.parametrize('action',('run_one','run_all','technical_smoke'))
def test_all_ordinary_entrypoints_refuse_technical_stop_before_loading(tmp_path,monkeypatch,action):
    (tmp_path/'technical_stop.json').write_text('{}')
    def forbidden(*args):raise AssertionError('processing started despite stop')
    monkeypatch.setattr(study,'_load',forbidden)
    with pytest.raises(RuntimeError,match='technical_stop'):
        getattr(study,action)(tmp_path,6) if action=='run_one' else getattr(study,action)(tmp_path)


def test_v2_stop_and_authorization_scope_are_enforced(tmp_path):
    (tmp_path/'execution_v2').mkdir();(tmp_path/'execution_v2/technical_stop.json').write_text('{}')
    with pytest.raises(RuntimeError,match='technical_stop'):check_authorization(tmp_path,0,None)
    (tmp_path/'evaluation_manifest.json').write_text('{}');(tmp_path/'new_protocol.md').write_text('Frozen scope')
    auth={'schema_version':1,'action':'authorized_continuation','execution_version':2,
          'evaluation_manifest_sha256':sha256(tmp_path/'evaluation_manifest.json'),
          'technical_stop_sha256':sha256(tmp_path/'execution_v2/technical_stop.json'),
          'protocol_path':'new_protocol.md','protocol_sha256':sha256(tmp_path/'new_protocol.md'),
          'allowed_indices':[6],'approved_by_user':True}
    path=tmp_path/'authorization.json';path.write_text(json.dumps(auth))
    check_authorization(tmp_path,6,path)
    # Retaining the original marker must not hide a newer v2 timeout.
    (tmp_path/'technical_stop.json').write_text('{"old":true}')
    check_authorization(tmp_path,6,path)
    (tmp_path/'execution_v2/technical_stop.json').write_text('{"new_timeout":true}')
    with pytest.raises(ValueError,match='scope'):check_authorization(tmp_path,6,path)
    (tmp_path/'execution_v2/technical_stop.json').write_text('{}')
    with pytest.raises(ValueError,match='scope'):check_authorization(tmp_path,8,path)
    (tmp_path/'new_protocol.md').write_text('changed')
    with pytest.raises(ValueError,match='protocol SHA'):check_authorization(tmp_path,6,path)


def test_supervisor_kills_slow_process_and_keeps_partial_evidence(tmp_path):
    marker=tmp_path/'partial.txt'
    script="import pathlib,time;pathlib.Path('partial.txt').write_text('checkpoint');print('started',flush=True);time.sleep(30)"
    result=bounded_process([sys.executable,'-c',script],cwd=tmp_path,timeout_s=.7)
    assert result['timed_out'] and result['returncode']!=0
    assert result['wall_s']<5
    assert marker.read_text()=='checkpoint' and 'started' in result['stdout']


def test_supervisor_reports_short_success_without_timeout(tmp_path):
    result=bounded_process([sys.executable,'-c',"print('complete')"],cwd=tmp_path,timeout_s=5)
    assert not result['timed_out'] and result['returncode']==0 and result['stdout'].strip()=='complete'


def _journals(tmp_path, epochs=(1.,2.)):
    names={}
    for kind in JOURNALS:
        name=f'{kind}_gcc_three.csv.gz';path=tmp_path/name
        rows=([{'processing_time_s':t,'valid':False,'failure_reason':'no_track'} for t in epochs]
              if kind=='tracking' else [])
        _write_csv_gz(path,rows,columns=list(rows[0]) if rows else EMPTY_COLUMNS[kind])
        names[name]=sha256(path)
    return names


def _verify(tmp_path, hashes):
    return verify_variant(tmp_path,'gcc','three',expected_epochs=[1.,2.],allowed_event_ids=['event-a'],expected_hashes=hashes)


def test_completion_accepts_empty_update_and_lifecycle_logs_but_requires_schedule(tmp_path):
    hashes=_journals(tmp_path);evidence=_verify(tmp_path,hashes)
    assert evidence['publication_count']==2
    context=completion_context(run_id='r',method='gcc',variant='three',bearing_sha='b',processing_sha='p',evaluation_sha='e')
    path=tmp_path/'completion.json';write_completion(path,context,evidence,origin='fixture')
    read_completion(path,context,tmp_path,expected_epochs=[1.,2.],allowed_event_ids=['event-a'])
    with pytest.raises(ValueError,match='provenance'):
        read_completion(path,{**context,'bearing_sha256':'other'},tmp_path,expected_epochs=[1.,2.],allowed_event_ids=['event-a'])


def test_one_csv_or_unclaimed_files_do_not_establish_completion(tmp_path):
    hashes=_journals(tmp_path)
    with pytest.raises(ValueError,match='missing claimed journal'):_verify(tmp_path,{'tracking_gcc_three.csv.gz':hashes['tracking_gcc_three.csv.gz']})
    (tmp_path/'hypotheses_gcc_three.csv.gz').unlink()
    with pytest.raises(ValueError,match='missing claimed journal'):_verify(tmp_path,hashes)


def test_complete_set_with_partial_schedule_is_rejected(tmp_path):
    hashes=_journals(tmp_path,epochs=(1.,))
    with pytest.raises(ValueError,match='schedule'):_verify(tmp_path,hashes)


def test_tampered_or_mixed_event_artifacts_are_rejected(tmp_path):
    hashes=_journals(tmp_path)
    path=tmp_path/'updates_gcc_three.csv.gz'
    _write_csv_gz(path,[{'event_id':'foreign','update_applied':False,'failure_reason':'bad'}])
    with pytest.raises(ValueError,match='SHA mismatch'):_verify(tmp_path,hashes)
    hashes[path.name]=sha256(path)
    with pytest.raises(ValueError,match='foreign event'):_verify(tmp_path,hashes)


def test_completed_and_partial_historical_variants_import_only_full_schedules():
    manifest=study._load(study.DEFAULT_OUTPUT)
    full=verified_historical_variant(study.DEFAULT_OUTPUT,manifest,manifest['specs'][6],
                                    'all_6_equal_gcc_wls','three_station_confirmation')
    assert full is not None and full[1]['publication_count']==177
    spec=manifest['specs'][11]
    results=[verified_historical_variant(study.DEFAULT_OUTPUT,manifest,spec,m,v) for m in
             ('all_6_equal_gcc_wls','equal_weight_srp_phat') for v in ('baseline','three_station_confirmation')]
    assert sum(r is not None for r in results)==3


def test_error_runs_break_on_invalid_samples_and_missing_time_and_use_exposure():
    result=large_error_runs([0.,.1,.2,.3,.6,.7],[11,11,np.nan,11,11,1],10)
    assert [r['frame_count'] for r in result]==[2,1,1]
    np.testing.assert_allclose([r['exposure_duration_s'] for r in result],[.2,.1,.1])
    assert safe_corr([1,1,1],[1,2,3]) is None
    with pytest.raises(ValueError,match='increasing'):large_error_runs([0,0],[11,11],10)


def test_bias_sign_and_R_quadratic_are_not_innovation_S():
    bias=np.array([.02,-.01]);e=bias+np.array([[.001,0],[-.001,0],[0,.002],[0,-.002]])
    r=np.diag([1e-6,4e-6]);result=calibration_diagnostics(e,bias,r)
    np.testing.assert_allclose(json.loads(result['mean_after_rad_json']),[0,0],atol=1e-17)
    assert result['d_R2_median_after']==pytest.approx(1)
    assert result['d_R2_median_before']>100
    assert result['fraction_inside_nominal_chi2_95']==1
    assert result['evaluation_used_for_calibration'] is False


def test_ideal_replays_preserve_source_events_R_validity_and_factual_times():
    import pandas as pd
    ledger=json.loads((DEFAULT_OUTPUT/'ideal_replays/replay_ledger.json').read_text())
    assert ledger['attempt_count']==4 and ledger['maximum_parallel_processes']==1
    manifest=study._load(study.DEFAULT_OUTPUT)
    for attempt in ledger['attempts']:
        directory=DEFAULT_OUTPUT/attempt['directory']
        summary=json.loads((directory/'summary.json').read_text())
        assert attempt['status']=='completed' and summary['executed_optimizations']==2
        assert summary['valid_publication_count']==168
        source=pd.read_csv(study._directory(study.DEFAULT_OUTPUT,manifest['specs'][attempt['index']])/'bearing_records.csv.gz')
        selected=source[(source.estimator_variant==attempt['method'])&(source.frame_index%32==0)]
        ideal=pd.read_csv(directory/'ideal_inputs.csv.gz')
        assert ideal.event_id.tolist()==selected.event_id.tolist()
        np.testing.assert_array_equal(ideal.reception_time_s,selected.frame_center_reception_time_s)
        np.testing.assert_array_equal(ideal.availability_time_s,selected.available_timestamp_s)
        assert ideal.valid.tolist()==selected.valid.tolist()
        assert all(json.loads(b)==[0.,0.] for b in ideal.bias_rad_json)
        for _,row in ideal.iterrows():
            calibration=next(c for c in manifest['processing']['calibration']['values']
                             if c['station_id']==row.station_id and c['estimator_variant']==attempt['method'])
            np.testing.assert_array_equal(json.loads(row.R_rad2_json),calibration['covariance_rad2'])


def test_empty_header_stub_is_not_a_verified_journal(tmp_path):
    hashes=_journals(tmp_path)
    path=tmp_path/'updates_gcc_three.csv.gz'
    _write_csv_gz(path,[],columns=('event_id',))
    hashes[path.name]=sha256(path)
    with pytest.raises(ValueError,match='schema'):_verify(tmp_path,hashes)


def test_modified_frozen_runner_does_not_disable_integrity(tmp_path,monkeypatch):
    path=tmp_path/'snapshot.py';path.write_text('changed runner')
    monkeypatch.setattr(study,'FROZEN_RUNNER_V1',path)
    with pytest.raises(ValueError,match='runner SHA mismatch'):study._load(study.DEFAULT_OUTPUT)


def test_stale_summary_is_rejected_when_verifying_historical_completion(tmp_path,monkeypatch):
    import shutil
    from analysis import unseen_execution_v2 as envelope
    manifest=study._load(study.DEFAULT_OUTPUT);spec=manifest['specs'][6]
    original=study._directory(study.DEFAULT_OUTPUT,spec)
    copied=tmp_path/'run';shutil.copytree(original,copied)
    summary=copied/'summary.json';summary.write_text('{}')
    monkeypatch.setattr(study,'_directory',lambda output,spec:copied)
    with pytest.raises(ValueError,match='historical artifact SHA mismatch: summary'):
        envelope.verified_historical_variant(tmp_path,manifest,spec,'all_6_equal_gcc_wls','baseline')


def test_authorized_worker_reuses_each_verified_completion_without_tracker(tmp_path,monkeypatch):
    """A copied fixture imports once, then skips all four verified variants."""
    import shutil
    import time
    from analysis import unseen_execution_v2 as envelope
    manifest=study._load(study.DEFAULT_OUTPUT);spec=manifest['specs'][6]
    original=study._directory(study.DEFAULT_OUTPUT,spec)
    copied=study._directory(tmp_path,spec);shutil.copytree(original,copied)
    shutil.copyfile(study.DEFAULT_OUTPUT/'evaluation_manifest.json',tmp_path/'evaluation_manifest.json')
    (tmp_path/'technical_stop.json').write_text('{}')
    (tmp_path/'protocol.md').write_text('Fixture-only continuation; no synthesis or tracking')
    auth={'schema_version':1,'action':'authorized_continuation','execution_version':2,
          'evaluation_manifest_sha256':sha256(tmp_path/'evaluation_manifest.json'),
          'technical_stop_sha256':sha256(tmp_path/'technical_stop.json'),
          'protocol_path':'protocol.md','protocol_sha256':sha256(tmp_path/'protocol.md'),
          'allowed_indices':[6],'approved_by_user':True}
    authorization=tmp_path/'authorization.json';authorization.write_text(json.dumps(auth))
    monkeypatch.setattr(study,'_load',lambda output:manifest)
    def forbidden(*args,**kwargs):raise AssertionError('verified variant was recomputed')
    monkeypatch.setattr(envelope,'_run_tracker',forbidden)
    monkeypatch.setattr(study,'_restore_bearings',forbidden)
    imported=envelope.worker(tmp_path,6,time.monotonic()+30,authorization)
    directory=tmp_path/imported['summary_directory']
    assert len(list(directory.glob('completion_*.json')))==4
    assert imported['tracker_replays_now']==0 and not imported['audio_synthesized_now']
    monkeypatch.setattr(envelope,'verified_historical_variant',forbidden)
    reused=envelope.worker(tmp_path,6,time.monotonic()+30,authorization)
    assert reused['tracker_replays_now']==0
    assert reused['method_variants']==imported['method_variants']
