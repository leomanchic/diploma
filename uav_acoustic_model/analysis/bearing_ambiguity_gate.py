"""Experimental opt-in gate. Only complete current frames / observables enter.

Kept outside the archived physics hash; feature code has its own frozen SHA.
No probability calibration is claimed. No source/truth/temporal model is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from estimators.gcc_phat import estimate_tdoas_gcc_phat
from model.geometry import all_pairs, baselines, direction_vector
from model.measurements import BearingMeasurement
from validation.moving_source_study import _delay_bounds

GCC = 'all_6_equal_gcc_wls'
SRP = 'equal_weight_srp_phat'
PHI, EL = np.meshgrid(np.deg2rad(np.arange(0,360,5)), np.deg2rad(np.arange(0,91,5)))
GRID = direction_vector(PHI.ravel(), EL.ravel())

@dataclass(frozen=True)
class ObservableFeatures:
    participation: float | None
    margin: float | None
    undefined_reason: str = ''

@dataclass(frozen=True)
class GateConfig:
    enabled: bool = False
    minimum_participation: float = 0.0
    minimum_margin: float = 0.0

    def __post_init__(self):
        for x in (self.minimum_participation, self.minimum_margin):
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError('gate thresholds must lie in [0,1]')


def participation(frame, fs):
    x = np.asarray(frame, float)
    if x.ndim != 2 or x.shape[1] < 3 or not np.all(np.isfinite(x)):
        return None, 'insufficient_spectral_information'
    if float(np.sqrt(np.mean(x*x))) <= 1e-10:
        return None, 'silence'
    f = np.fft.rfftfreq(x.shape[1], 1/float(fs))
    band = (f >= 200) & (f <= 10000)
    if np.count_nonzero(band) < 3:
        return None, 'insufficient_spectral_information'
    power = np.sum(np.abs(np.fft.rfft((x-x.mean(axis=1,keepdims=True))*np.hanning(x.shape[1]),axis=1))**2,axis=0)[band]
    denom = len(power)*float(power@power)
    if not np.isfinite(denom) or denom <= 0:
        return None, 'undefined_spectrum'
    return float(np.clip(power.sum()**2/denom,0,1)), ''


def separated_margin(values, peak_value, competitor_mask):
    values = np.asarray(values,float)
    if not np.all(np.isfinite(values)) or not np.isfinite(peak_value):
        return None, None, 'nonfinite_score'
    if not np.any(competitor_mask):
        return None, None, 'missing_competitor'
    scale = float(np.ptp(values))
    if scale <= 1e-12:
        return None, None, 'undefined_score'
    selected = np.flatnonzero(competitor_mask)
    second = int(selected[np.argmax(values[selected])])
    return float(np.clip((peak_value-values[second])/scale,0,1)), second, ''


def frame_features(frame, positions, fs, srp_direction):
    """Truth-free diagnostics using the same lag model as existing SRP.

    Returns two scalar features per method and inspectable current-frame curves.
    Direction is the current raw SRP estimate, never an expected direction.
    """
    p, reason = participation(frame,fs)
    if reason:
        return {m:ObservableFeatures(p,None,reason) for m in (GCC,SRP)}, {}
    pairs=all_pairs(len(positions))
    _, diagnostics = estimate_tdoas_gcc_phat(frame,fs,pairs,
        maximum_delay_seconds=_delay_bounds(positions,pairs,fs),interpolation_factor=2,
        minimum_frequency_hz=200,maximum_frequency_hz=10000,relative_spectral_floor=1e-8)
    margins=[]; peaks=[]; competitors=[]; reasons=[]
    curves={}
    for k,item in enumerate(diagnostics):
        curves[f'lag_{k}']=item.lags_samples
        curves[f'correlation_{k}']=item.correlation
        if item.invalid:
            return {m:ObservableFeatures(p,None,'invalid_pair:'+str(item.invalid_reason)) for m in (GCC,SRP)},curves
        peak=int(np.argmax(item.correlation))
        margin,comp,r=separated_margin(item.correlation,item.correlation[peak],np.abs(item.lags_samples-item.lags_samples[peak])>=2)
        margins.append(margin); peaks.append(peak); competitors.append(-1 if comp is None else comp); reasons.append(r)
    curves['gcc_peak_indices']=np.asarray(peaks)
    curves['gcc_competitor_indices']=np.asarray(competitors)
    gcc=ObservableFeatures(p,min(margins) if not any(reasons) else None,next((r for r in reasons if r),''))
    splines=[CubicSpline(d.lags_samples,d.correlation,extrapolate=False) for d in diagnostics]
    def scores(v):
        delays=np.asarray(v)@baselines(positions,pairs).T/343*fs
        return sum(s(delays[:,k]) for k,s in enumerate(splines))/len(pairs)
    values=scores(GRID)
    u=np.asarray(srp_direction,float)
    if u.shape!=(3,) or not np.all(np.isfinite(u)):
        srp=ObservableFeatures(p,None,'invalid_direction'); comp=None; peak=float('nan')
    else:
        peak=float(scores(u[None,:])[0])
        margin,comp,r=separated_margin(values,peak,GRID@u<=np.cos(np.deg2rad(20)))
        srp=ObservableFeatures(p,margin,r)
    curves.update(srp_directions=GRID,srp_scores=values,srp_selected_direction=u,
                  srp_selected_score=np.asarray(peak),srp_competitor_index=np.asarray(-1 if comp is None else comp))
    return {GCC:gcc,SRP:srp},curves


def decision(features: ObservableFeatures, config: GateConfig):
    if not config.enabled:
        return True, ''
    if features.undefined_reason or features.participation is None or features.margin is None:
        return False,'ambiguity_gate:'+ (features.undefined_reason or 'undefined_score')
    if not np.isfinite(features.participation) or not np.isfinite(features.margin):
        return False,'ambiguity_gate:undefined_score'
    if features.participation < config.minimum_participation:
        return False,'ambiguity_gate:low_spectral_participation'
    if features.margin < config.minimum_margin:
        return False,'ambiguity_gate:competing_peak'
    return True,''


def apply_gate(measurement: BearingMeasurement, features: ObservableFeatures, config: GateConfig):
    """Opt-out is object identity; opt-in exclusion uses existing invalid contract."""
    if not config.enabled or not measurement.valid:
        return measurement
    accepted, reason=decision(features,config)
    if accepted:
        return measurement
    return BearingMeasurement.invalid(station_id=measurement.station_id,
        sequence_id=measurement.sequence_id,frame_index=measurement.frame_index,
        reception_center_timestamp_s=measurement.reception_center_timestamp_s,
        available_timestamp_s=measurement.available_timestamp_s,
        estimator_variant=measurement.estimator_variant,invalid_reason=reason,
        quality_metadata=measurement.quality_metadata)
