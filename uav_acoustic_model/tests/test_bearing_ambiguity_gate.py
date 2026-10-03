import inspect
import json
from dataclasses import replace
import numpy as np
import pytest
from analysis.bearing_ambiguity_gate import (GateConfig,ObservableFeatures,apply_gate,decision,frame_features,participation,separated_margin)
from analysis.bearing_ambiguity_study import protocol
from model.measurements import BearingMeasurement
from model.bearing_events import CausalBearingEventStream,bearing_event_id


def measurement(i=0):
    return BearingMeasurement(station_id='S0',sequence_id='gate-test',frame_index=i,reception_center_timestamp_s=i+.01,
        available_timestamp_s=i+.03,direction_local=[1,0,0],covariance_tangent_rad2=np.eye(2)*1e-4,
        calibration_bias_tangent_rad=[0,0],estimator_variant='all_6_equal_gcc_wls')


def test_disabled_is_exact_original_even_undefined():
    raw=measurement();assert apply_gate(raw,ObservableFeatures(None,None,'silence'),GateConfig()) is raw


def test_rejected_identity_reason_and_causal_eligible_set():
    raw=measurement();bad=apply_gate(raw,ObservableFeatures(.01,.5),GateConfig(True,.1,.2))
    assert not bad.valid and bad.invalid_reason=='ambiguity_gate:low_spectral_participation'
    assert bearing_event_id(raw)==bearing_event_id(bad)
    processor=CausalBearingEventStream([bad,measurement(1)],estimator_variant=raw.estimator_variant)
    prefix=processor.advance_to(2)
    assert len(prefix.measurements)==1
    assert prefix.measurements[0].frame_index==1


def test_truth_and_future_absent_contract():
    assert list(inspect.signature(frame_features).parameters)==['frame','positions','fs','srp_direction']
    assert list(inspect.signature(decision).parameters)==['features','config']
    assert set(ObservableFeatures.__dataclass_fields__)=={'participation','margin','undefined_reason'}
    x=measurement();good=ObservableFeatures(.5,.8)
    assert apply_gate(x,good,GateConfig(True,.1,.2)) is x


def test_silence_insufficient_and_undefined():
    assert participation(np.zeros((4,1024)),48000)==(None,'silence')
    assert participation(np.ones((4,4)),48000)[1]=='insufficient_spectral_information'
    assert separated_margin(np.ones(4),1,np.array([False,False,True,True]))[2]=='undefined_score'
    assert decision(ObservableFeatures(None,None,'silence'),GateConfig(True))[1]=='ambiguity_gate:silence'
    assert decision(ObservableFeatures(.5,float('nan')),GateConfig(True))[1]=='ambiguity_gate:undefined_score'


def test_margin_and_spectral_invariances():
    x=np.random.default_rng(1).normal(size=(4,1024))
    assert participation(x,48000)[0]==pytest.approx(participation(7*x+2,48000)[0],abs=1e-14)
    v=np.array([0,1,.4,.6])
    assert separated_margin(v,1,np.array([1,0,0,1],bool))[0]==pytest.approx(.4)
    assert separated_margin(4*v+3,7,np.array([1,0,0,1],bool))[0]==pytest.approx(.4)


def test_frozen_split_and_limits():
    p=protocol();dev=[s for s in p['specs'] if s['split']=='development'];ev=[s for s in p['specs'] if s['split']=='evaluation']
    assert len(dev)==len(ev)==8
    assert {s['trajectory'] for s in dev}=={'spatial'}
    assert {s['trajectory'] for s in ev}=={'radial'}
    assert {s['source_seed'] for s in dev}.isdisjoint({s['source_seed'] for s in ev})
    assert {s['noise_seed'] for s in dev}.isdisjoint({s['noise_seed'] for s in ev})
    assert p['harmonic']['development']!=p['harmonic']['evaluation']
    assert p['limits']==dict(audio_streams=16,tracker_replays=8,audio_timeout_s=1200,replay_timeout_s=600,total_worker_wall_s=21600)


def test_reasons_counts_and_good_denominator():
    from analysis.bearing_ambiguity_report import stats
    rows=[dict(event_id=str(i),valid=True,geodesic_error_deg=e,frame_start_reception_time_s=i,frame_end_reception_time_s=i+.1) for i,e in enumerate([1,20,40])]
    f={str(i):ObservableFeatures(p,.9) for i,p in enumerate([.8,.01,.01])}
    s=stats(rows,f,GateConfig(True,.1,0))
    assert s['accepted']==1 and s['rejected']==2
    assert s['over_10_passed_all_valid']==0 and s['good_false_rejection_conditional']==0
    assert sum(json.loads(s['reasons_json']).values())==s['rejected']
    assert s['missing_series_max_s']==pytest.approx(1.1)


def test_raw_estimates_unchanged_by_feature_extraction():
    from validation.moving_source_study import estimate_independent_frame
    from model.geometry import comparison_arrays
    positions=comparison_arrays()['tetrahedral']
    frame=np.random.default_rng(52).normal(size=(4,1024))
    original=estimate_independent_frame(frame,positions,48000)
    copy=frame.copy()
    features,curves=frame_features(frame,positions,48000,original['equal_weight_srp_phat']['direction'])
    again=estimate_independent_frame(frame,positions,48000)
    assert np.array_equal(frame,copy)
    for method in ('all_6_equal_gcc_wls','equal_weight_srp_phat'):
        assert np.array_equal(original[method]['direction'],again[method]['direction'])
        assert original[method]['valid']==again[method]['valid']
    assert len([k for k in curves if k.startswith('correlation_')])==6
    assert curves['srp_scores'].shape==(1368,)


def test_rejected_excluded_from_tracker_all_stages():
    from estimators.retarded_ekf_manoeuvre import CausalManoeuvreRetardedTimeEKF,ManoeuvreHistoryConfig
    from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
    from validation.three_station_audio_tracking_study import pilot_stations
    bad=[]
    for i in range(12):
        x=replace(measurement(i),station_id=f'S{i%3}')
        bad.append(apply_gate(x,ObservableFeatures(.01,.1),GateConfig(True,.5,.5)))
    ekf=CausalManoeuvreRetardedTimeEKF(pilot_stations(),bad,estimator_variant=bad[0].estimator_variant,
        history_config=ManoeuvreHistoryConfig(np.eye(3)),recovery_config=InitializationRecoveryConfig(confirmation_station_count=3))
    final=ekf.advance_to(20)
    assert not final.valid
    assert not ekf.batch_fit_diagnostics and not final.event_uses and not final.update_diagnostics


def test_actual_recording_keys_and_scenario_support():
    from analysis.bearing_ambiguity_study import recording_key,scenario
    from analysis.unseen_manoeuvres_and_sources import recording_inputs,_frozen_protocol
    p=protocol();m={**p,'recordings':recording_inputs(_frozen_protocol())}
    assert set(recording_key(s) for s in p['specs'])==set(m['recordings'])
    for index in (0,8):
        stations,trajectory,offset,info=scenario(m,p['specs'][index])
        assert info['reception_start_s']==0
        assert len(stations)==3 and np.all(np.isfinite(trajectory.q(0)))
        assert trajectory.maximum_speed_mps<343


def test_complete_checkpoint_rejects_substitution(tmp_path):
    from analysis.bearing_ambiguity_study import complete,verify_completion
    (tmp_path/'data.txt').write_text('recorded')
    context=dict(kind='artifact',index=0,study_manifest_sha256='test')
    complete(tmp_path,context);verify_completion(tmp_path,context)
    (tmp_path/'data.txt').write_text('substituted')
    with pytest.raises(ValueError,match='SHA mismatch'):verify_completion(tmp_path,context)


def test_selection_uses_development_only_and_fixed_grid():
    from analysis.bearing_ambiguity_report import select_thresholds
    methods=('all_6_equal_gcc_wls','equal_weight_srp_phat')
    rows=[dict(event_id=str(i),valid=True,geodesic_error_deg=e,frame_start_reception_time_s=i,frame_end_reception_time_s=i+.1) for i,e in enumerate([1]*8+[40]*2)]
    f={str(i):ObservableFeatures(.8 if i<8 else .01,.8) for i in range(10)}
    groups=[(dict(source_class=cls),method,'S0',rows,f) for cls in ('broadband','nonstationary_harmonic') for method in methods]
    thresholds,tradeoff=select_thresholds(groups,dict(participation=[0,.1],margin=[0]))
    assert all(c['minimum_participation']==.1 for c in thresholds.values())
    assert len(tradeoff)==4


def test_published_failure_loads_without_restarting():
    from analysis.bearing_ambiguity_failure import audit_stop
    r=audit_stop()
    assert r['worker_attempts']==1
    assert r['actual_audio_streams']==r['actual_replays']==0
    assert not r['thresholds_selected'] and not r['evaluation_accessed']


def test_stopped_study_cannot_resume():
    from analysis.bearing_ambiguity_study import execute,OUT
    with pytest.raises((ValueError,RuntimeError)):
        execute(OUT,'development')
    assert not (OUT/'worker.lock').exists()


def test_audio_worker_contract_without_resynthesis(monkeypatch,tmp_path):
    """Inject a short stream fixture, exercise frame/serialization/completion APIs."""
    from types import SimpleNamespace
    from analysis import bearing_ambiguity_study as runner
    from validation.gazebo_experiment import stations_from_experiment,sha256
    from simulation.trajectory import StationaryTrajectory
    p=protocol();spec=p['specs'][0];stations=stations_from_experiment({'processing':p['processing']})
    fs=p['processing']['audio']['sampling_rate_hz'];samples=1536
    x=np.random.default_rng(61).normal(size=(4,samples))
    stream=SimpleNamespace(sampling_rate_hz=fs,reception_times_s=np.arange(samples)/fs,source_seed=spec['source_seed'],
        stations=[SimpleNamespace(station_id=s.station_id,channels=x,noise=x,noise_seed=i+10,effective_snr_db=1) for i,s in enumerate(stations)])
    m={**p,'source_sha256':runner.source_hashes()}
    (tmp_path/'study_manifest.json').write_text(json.dumps(m))
    monkeypatch.setattr(runner,'load',lambda output:m)
    monkeypatch.setattr(runner,'source_bank',lambda *args:(np.ones(samples),dict(start_time_s=-2,signal_sha256='fixture')))
    monkeypatch.setattr(runner,'scenario',lambda *args:(stations,StationaryTrajectory([200,10,80]),np.zeros(3),dict(duration_s=samples/fs,time_origin_gazebo_s=0,state_sha256='fixture')))
    monkeypatch.setattr(runner,'synthesize_multistation_audio',lambda *args,**kwargs:stream)
    runner.audio_worker(tmp_path,0)
    c=runner.verify_completion(runner.run_dir(tmp_path,0),dict(kind='audio',index=0,study_manifest_sha256=sha256(tmp_path/'study_manifest.json')))
    assert c['status']=='complete_verified'
    assert len(runner._measurement_rows(runner.run_dir(tmp_path,0)/'bearing_records.csv.gz'))==12


def test_absent_good_labels_are_explicit_development_failure():
    from analysis.bearing_ambiguity_report import select_thresholds
    from analysis.bearing_ambiguity_study import METHODS
    rows=[dict(event_id='x',valid=True,geodesic_error_deg=90,frame_start_reception_time_s=0,frame_end_reception_time_s=.1)]
    f={'x':ObservableFeatures(.5,.9)}
    groups=[(dict(source_class=cls),method,'S0',rows,f) for method in METHODS for cls in ('broadband','nonstationary_harmonic')]
    thresholds,candidates=select_thresholds(groups,dict(participation=[0,.1],margin=[0]))
    assert all(t==dict(minimum_participation=0,minimum_margin=0) for t in thresholds.values())
    assert not any(c['feasible'] for c in candidates)
    assert all(json.loads(c['good_false_rejection_by_class_json'])==dict(broadband=None,nonstationary_harmonic=None) for c in candidates)


def test_confirmation_availability_excludes_future_delivery():
    from analysis.bearing_ambiguity_report import availability
    rows=[dict(event_id=str(i),valid=True,station_id=f'S{i}',frame_index=0,available_timestamp_s=t) for i,t in enumerate([.1,.2,.4])]
    features={str(i):ObservableFeatures(.9,.9) for i in range(3)}
    r=availability(rows,features,GateConfig(True))
    assert r['trailing_3s_3_stations_fraction']==pytest.approx(1/3)
    assert r['trailing_3s_2_stations_fraction']==pytest.approx(2/3)


def test_evaluation_worker_requires_gate_freeze(monkeypatch,tmp_path):
    from analysis import bearing_ambiguity_study as runner
    monkeypatch.setattr(runner,'load',lambda output:protocol())
    with pytest.raises(FileNotFoundError):runner.audio_worker(tmp_path,8)
    assert not (tmp_path/'runs').exists() and not (tmp_path/'source_banks').exists()


def test_freeze_refuses_evaluation_access(monkeypatch,tmp_path):
    from analysis import bearing_ambiguity_report as report
    monkeypatch.setattr(report,'load',lambda output:protocol())
    p=tmp_path/'runs'/'08';p.mkdir(parents=True);(p/'checkpoint.json').write_text('{}')
    with pytest.raises(ValueError,match='evaluation accessed before gate freeze'):report.freeze(tmp_path)
    assert not (tmp_path/'gate_manifest.json').exists()


def test_published_retry_complete_and_rejections_excluded_all_roles():
    from pathlib import Path
    from analysis.bearing_ambiguity_study import ROOT
    from analysis.bearing_ambiguity_report import verify
    from analysis.localization_error_attribution import _read_csv_gz
    out=ROOT/'results/bearing_ambiguity_retry_001'
    assert verify(out)==(16,8)
    for directory in (out/'replays').iterdir():
        rejected={r['event_id'] for r in _read_csv_gz(directory/'gate_events.csv.gz') if r['gate_rejected']=='True'}
        method=json.loads((directory/'summary.json').read_text())['method']
        for kind in ('batch_fits','hypotheses','event_uses','updates'):
            rows=_read_csv_gz(directory/f'{kind}_{method}_three_station_confirmation.csv.gz')
            for r in rows:
                if r.get('event_id'):assert r['event_id'] not in rejected
                for key in ('event_ids','construction_event_ids','confirmation_event_ids'):
                    if r.get(key):assert set(json.loads(r[key])).isdisjoint(rejected)


def test_published_gcc_opt_out_numeric_parity():
    from analysis.bearing_ambiguity_study import ROOT
    from analysis.localization_error_attribution import _read_csv_gz
    method='all_6_equal_gcc_wls'
    for index in (8,12):
        root=ROOT/'results/bearing_ambiguity_retry_001/replays'
        for journal in ('tracking','updates','batch_fits','hypotheses','event_uses','lifecycle'):
            name=f'{journal}_{method}_three_station_confirmation.csv.gz'
            assert _read_csv_gz(root/f'{index:02d}_{method}_off'/name)==_read_csv_gz(root/f'{index:02d}_{method}_on'/name)
