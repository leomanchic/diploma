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
