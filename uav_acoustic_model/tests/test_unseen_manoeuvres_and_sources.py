"""Transfer-study invariants without audio synthesis or tracker execution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.unseen_manoeuvres_and_sources import (
    _frozen_protocol, _run_identity, nonstationary_harmonic_signal,
    paired_shared, recording_inputs, summarize_track,
)


def _track(t, *, valid, error=1.0, covered=True, final_fit=0):
    return {
        'processing_time_s':t,'valid':valid,'confirmed':valid,
        'status':'confirmed' if valid else 'initializing','failure_reason':'',
        'position_covariance_m2_json':json.dumps(np.eye(3).tolist()) if valid else '',
        'position_enu_m_json':json.dumps([0,0,0]) if valid else '',
        'position_error_m':error if valid else '',
        'velocity_error_mps':2.0 if valid else '',
        'position_nees':1.0 if valid else '',
        'position_nominal_95_covered':covered if valid else '',
        'batch_optimization_count':final_fit,'reset_count':0,
    }


def test_recordings_are_two_new_physical_flights_with_kinematic_checks():
    frozen=_frozen_protocol()
    info=recording_inputs(frozen)
    assert set(info)=={'spatial_manoeuvre','radial_approach_depart'}
    assert all(x['export_rate_hz']==50 for x in info.values())
    assert all(x['duration_s']>15 for x in info.values())
    assert len({x['state_sha256'] for x in info.values()})==2
    root=Path(__file__).resolve().parents[1]
    for name,item in info.items():
        assessment=json.loads((root/item['path']/'export_assessment.json').read_text())
        assert assessment['recording_sha256']==item['state_sha256']
        assert assessment['maximum_gap_s']<=.020001
        assert assessment['velocity_error_rms_50hz_mps']<.02


def test_harmonic_bank_is_nonstationary_unit_rms_and_seeded():
    a=nonstationary_harmonic_signal(48_000,48_000,123)
    b=nonstationary_harmonic_signal(48_000,48_000,123)
    c=nonstationary_harmonic_signal(48_000,48_000,124)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
    assert np.isclose(np.sqrt(np.mean(a*a)),1.0,atol=1e-12)
    assert np.all(np.isfinite(a))
    assert abs(np.std(a[:12_000])-np.std(a[24_000:36_000]))>.01


def test_run_identity_ignores_paths_but_tracks_recording_source_and_seed():
    frozen=_frozen_protocol()
    info={'state_sha256':'a','path':'machine_a/flight'}
    bank={'signal_sha256':'b','path':'machine_a/source.npy'}
    spec={'source_class':'broadband','distance_m':700,'replicate':1,
          'source_seed':11,'noise_seed':22,'snr_ref_db':10.0}
    reference=_run_identity(frozen,info,bank,spec)
    assert reference==_run_identity(frozen,{**info,'path':'machine_b/flight'},
                                    {**bank,'path':'machine_b/source.npy'},spec)
    assert reference!=_run_identity(frozen,{**info,'state_sha256':'c'},bank,spec)
    assert reference!=_run_identity(frozen,info,{**bank,'signal_sha256':'d'},spec)
    assert reference!=_run_identity(frozen,info,bank,{**spec,'noise_seed':23})


def test_denominators_failures_and_before_fit_budget_are_retained():
    tracks=[_track(1.0,valid=False,final_fit=1),_track(2.0,valid=True,final_fit=1)]
    fits=[{'generation':1,'reason':'batch_passed','fit_count_after_attempt':1},
          {'generation':1,'reason':'computational_budget_exceeded_before_fit',
           'fit_count_after_attempt':1}]
    result=summarize_track(tracks,[],fits,[],reception_start_s=0)
    assert result['publication_count']==2
    assert result['valid_publication_count']==1
    assert result['availability_fraction']==.5
    assert result['total_executed_optimizations']==1
    assert result['attempts_rejected_before_optimization']==1
    assert result['nominal_5m_count']==1
    assert result['nominal_5m_false_precision_fraction']==0
    assert not result['severe_first_confirmation_over_50m']
    missing=summarize_track([_track(1.0,valid=False)],[],[],[],reception_start_s=0)
    assert missing['publication_count']==1 and missing['availability_fraction']==0
    assert missing['position_rmse_m_conditional'] is None
    assert missing['nominal_5m_false_precision_fraction'] is None
    assert not missing['ever_confirmed']


def test_shared_comparison_handles_disjoint_valid_times_and_stable_schedule():
    baseline=[_track(1.0,valid=True),_track(2.0,valid=False)]
    improved=[_track(1.0,valid=False),_track(2.0,valid=True)]
    result=paired_shared(baseline,improved)
    assert result['shared_valid_count']==0
    assert result['baseline_only_valid_count']==1
    assert result['new_only_valid_count']==1
    assert result['baseline_shared_coverage'] is None
    with pytest.raises(ValueError,match='paired publication times'):
        paired_shared(baseline,improved[:1])


def test_rebased_processing_time_preserves_physical_gazebo_time_and_derivatives():
    from analysis.unseen_manoeuvres_and_sources import RebasedTrajectory

    class Linear:
        knot_times_s=np.array([100.0,120.0])
        maximum_speed_mps=2.0
        kind='recorded_test'
        def q(self,t):
            v=np.asarray(t,dtype=float)
            return np.stack((v,2*v,0*v),axis=-1)
        def v(self,t):
            v=np.asarray(t,dtype=float)
            return np.broadcast_to(np.array([1.0,2.0,0.0]),v.shape+(3,))
        def a(self,t):
            return np.zeros(np.asarray(t).shape+(3,))

    original=Linear()
    local=RebasedTrajectory(original,105.0)
    np.testing.assert_allclose(local.knot_times_s,[-5.0,15.0])
    for local_time in (0.0,2.5,10.0):
        np.testing.assert_allclose(local.q(local_time),original.q(local_time+105.0))
        np.testing.assert_allclose(local.v(local_time),original.v(local_time+105.0))
        np.testing.assert_allclose(local.a(local_time),original.a(local_time+105.0))
