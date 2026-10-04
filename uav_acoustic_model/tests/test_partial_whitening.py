"""Independent spectral references and bounded experiment contracts."""
from dataclasses import fields
import numpy as np
import pytest
from scipy.fft import next_fast_len
from scipy.interpolate import CubicSpline
from analysis.partial_whitening_frontend import (
    BETAS,METHODS,additional_weights,prepare_spectrum,gcc_partial,
    estimate_frame,estimate_tdoas_partial)
from estimators.gcc_phat import gcc_phat
from validation.moving_source_study import estimate_independent_frame,_delay_bounds
from model.geometry import comparison_arrays,all_pairs,baselines,direction_vector
from simulation.fractional_delay import frequency_domain_delay
from simulation.signals import deterministic_bandlimited_signal
FS=48000
KW=dict(maximum_delay_seconds=12.3/FS,interpolation_factor=2,
        minimum_frequency_hz=200,maximum_frequency_hz=10000,relative_spectral_floor=1e-8)


def spectral_reference(x,y,beta,fs=FS,low=200,high=10000,floor=1e-8):
    # Does not call the implementation's spectrum/weight preparation.
    n=next_fast_len(len(x)+len(y));f=np.fft.rfftfreq(n,1/fs)
    c=np.fft.rfft(x,n)*np.conj(np.fft.rfft(y,n));m=np.abs(c)
    band=(f>=low)&(f<=high);active=band&(m>floor*np.max(m[band]))
    p=np.zeros_like(c);p[active]=c[active]/m[active]
    w=np.ones_like(m) if beta==1 else np.where(band,(m/np.max(m[band]))**(1-beta),0)
    factors=np.full(len(f),2.);factors[0]=1
    if n%2==0:factors[-1]=1
    return n,f,p*w,factors


def direct_values(x,y,beta,delays_samples):
    n,f,p,w=spectral_reference(x,y,beta)
    return np.real(np.sum(w[:,None]*p[:,None]*np.exp(2j*np.pi*f[:,None]*np.asarray(delays_samples)[None,:]/FS),axis=0))/n


def test_beta_one_exact_baseline_all_fields_and_zero_bins():
    x=np.random.default_rng(412).normal(size=1024);y=np.roll(x,3)
    a=gcc_phat(x,y,FS,**KW);b=gcc_partial(x,y,FS,beta=1,**KW)
    for field in fields(a):
        np.testing.assert_equal(getattr(a,field.name),getattr(b,field.name))
    np.testing.assert_array_equal(additional_weights([0,2,0],[True]*3,1),[1,1,1])
    positions=comparison_arrays()['tetrahedral'];frame=np.random.default_rng(4).normal(size=(4,1024))
    a=estimate_independent_frame(frame,positions,FS);b=estimate_frame(frame,positions,FS,beta=1)
    for m in METHODS:
        for key in a[m]:
            if 'runtime' not in key:np.testing.assert_equal(a[m][key],b[m][key])


@pytest.mark.parametrize('beta',BETAS)
def test_fft_matches_independent_sum_and_delay_sign(beta):
    signal=deterministic_bandlimited_signal(FS,.05,minimum_frequency_hz=300,maximum_frequency_hz=10000,tone_count=31)
    x=frequency_domain_delay(signal,3.5,output_length=len(signal)+8)
    y=frequency_domain_delay(signal,.25,output_length=len(signal)+8)
    r=gcc_partial(x,y,FS,beta=beta,**{**KW,'interpolation_factor':32})
    np.testing.assert_allclose(r.correlation,direct_values(x,y,beta,r.lags_samples),atol=2e-14,rtol=2e-13)
    assert r.delay_samples==pytest.approx(3.25,abs=2e-4)
    reverse=gcc_partial(y,x,FS,beta=beta,**{**KW,'interpolation_factor':32})
    assert reverse.delay_samples==pytest.approx(-r.delay_samples,abs=1e-10)
    assert abs(r.delay_samples)<=12.3


@pytest.mark.parametrize('beta',BETAS)
def test_srp_cubic_matches_direct_weighted_spatial_sum(beta):
    positions=comparison_arrays()['tetrahedral'];pairs=all_pairs(4)
    frame=np.random.default_rng(51).normal(size=(4,1024))
    _,ds=estimate_tdoas_partial(frame,FS,pairs,beta=beta,**{**KW,'maximum_delay_seconds':_delay_bounds(positions,pairs,FS)})
    directions=direction_vector(np.deg2rad(np.arange(0,360,15)),np.deg2rad(30))
    delays=directions@baselines(positions,pairs).T/343*FS
    cubic=sum(CubicSpline(d.lags_samples,d.correlation,extrapolate=False)(delays[:,k]) for k,d in enumerate(ds))/6
    reference=sum(direct_values(frame[i],frame[j],beta,delays[:,k]) for k,(i,j) in enumerate(pairs))/6
    # Fourth-order interpolation bound for a half-sample cubic grid.
    bound=0
    for i,j in pairs:
        n,f,p,w=spectral_reference(frame[i],frame[j],beta)
        bound+=5/384*.5**4*np.sum(w*np.abs(p)*(2*np.pi*f/FS)**4)/n/6
    assert np.max(np.abs(cubic-reference))<2*bound


@pytest.mark.parametrize('beta',BETAS)
def test_amplitude_invariance_above_existing_floor(beta):
    frame=np.random.default_rng(77).normal(size=(4,1024));pos=comparison_arrays()['tetrahedral']
    a=estimate_frame(frame,pos,FS,beta=beta);b=estimate_frame(frame*np.array([.25,2,5,.7])[:,None],pos,FS,beta=beta)
    for m in METHODS:
        assert a[m]['valid']==b[m]['valid']
        np.testing.assert_allclose(a[m]['direction'],b[m]['direction'],atol=2e-5,rtol=0)


@pytest.mark.parametrize('beta',BETAS)
def test_silence_zero_band_and_tone_ambiguity(beta):
    r=gcc_partial(np.zeros(1024),np.zeros(1024),FS,beta=beta,**KW)
    assert r.invalid and np.isnan(r.delay_samples) and r.invalid_reason=='signal_energy_below_threshold'
    r=gcc_partial(np.ones(1024),np.ones(1024),FS,beta=beta,minimum_signal_rms=0,
        **{**KW,'minimum_frequency_hz':23990,'maximum_frequency_hz':24000})
    assert r.invalid and r.invalid_reason=='spectral_energy_below_threshold'
    # A clean 8 kHz tone repeats every six samples; it is not a unique DOA test.
    t=np.arange(2048)/FS;x=np.cos(2*np.pi*8000*t);y=np.cos(2*np.pi*8000*(t-1.5/FS))
    r=gcc_partial(x,y,FS,beta=beta,**{**KW,'minimum_frequency_hz':7900,'maximum_frequency_hz':8100})
    assert not r.invalid and np.all(np.isfinite(r.correlation))
    peaks=[r.correlation[np.argmin(abs(r.lags_samples-lag))] for lag in (-7.5,-1.5,4.5,10.5)]
    assert sum(v>.8*max(peaks) for v in peaks)>=2


def test_beta_grid_is_closed():
    with pytest.raises(ValueError,match='beta'):gcc_partial([1,2],[1,2],FS,beta=.6,**KW)


def test_frozen_protocol_seeds_settings_and_bounds():
    from analysis.partial_whitening_study import protocol,scenario,recording_key
    from analysis.unseen_manoeuvres_and_sources import recording_inputs,_frozen_protocol
    p=protocol();dev=p['specs'][:8];ev=p['specs'][8:]
    assert len(dev)==len(ev)==8
    assert {s['source_seed'] for s in dev}.isdisjoint(s['source_seed'] for s in ev)
    assert {s['noise_seed'] for s in dev}.isdisjoint(s['noise_seed'] for s in ev)
    assert len({s['noise_seed'] for s in p['specs']})==16
    assert p['harmonic']['development']!=p['harmonic']['evaluation']
    assert p['limits']==dict(audio_streams=16,tracker_replays=8,audio_timeout_s=1200,replay_timeout_s=600,total_worker_wall_s=21600)
    m={**p,'recordings':recording_inputs(_frozen_protocol())}
    for index in (0,8):
        stations,trajectory,offset,info=scenario(m,p['specs'][index])
        assert recording_key(p['specs'][index]) in m['recordings']
        assert len(stations)==3 and np.isfinite(trajectory.q(0)).all() and trajectory.maximum_speed_mps<343


def test_development_selection_constraints_ties_and_invalid_loss():
    from analysis.partial_whitening_report import select_beta,stats
    rows=[dict(valid=v,geodesic_error_deg=e,invalid_reason='invalid' if not v else '',frame_index=i,frame_start_reception_time_s=i,frame_end_reception_time_s=i+.1) for i,(v,e) in enumerate([(True,1),(True,20),(False,float('nan'))])]
    assert stats(rows)['loss']==pytest.approx(2/3)
    cells=[]
    for m in METHODS:
        for b in BETAS:
            for cls in ('broadband','nonstationary_harmonic'):
                cells.append(dict(method=m,beta=b,source_class=cls,sampling='all',loss=.4 if cls!='broadband' else (.03 if b==.5 else 0),valid_fraction=1,median_deg=1,p95_deg=2,over_5_conditional=0,over_10_conditional=0,over_30_conditional=0))
    chosen,trade=select_beta(cells);assert all(b==1 for b in chosen.values())
    for c in cells:
        if c['beta'] in (.75,.5) and c['source_class']=='nonstationary_harmonic':c['loss']=.2
    chosen,trade=select_beta(cells);assert all(b==.75 for b in chosen.values())
    assert all(not r['feasible'] for r in trade if r['beta']==.5)
    # Invalid/undefined conditional accuracy must not qualify a new variant.
    for c in cells:
        if c['beta']==.75 and c['source_class']=='broadband':c['p95_deg']=None
    chosen,_=select_beta(cells);assert all(b==1 for b in chosen.values())


def test_worker_serialization_one_shared_stream_and_disjoint_beta_ids(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from analysis import partial_whitening_study as runner
    from validation.gazebo_experiment import stations_from_experiment
    from simulation.trajectory import StationaryTrajectory
    p=runner.protocol();spec=p['specs'][0];stations=stations_from_experiment({'processing':p['processing']})
    samples=1536;fs=p['processing']['audio']['sampling_rate_hz'];x=np.random.default_rng(613).normal(size=(4,samples))
    stream=SimpleNamespace(sampling_rate_hz=fs,reception_times_s=np.arange(samples)/fs,source_seed=spec['source_seed'],stations=[SimpleNamespace(station_id=s.station_id,channels=x,noise=x,noise_seed=spec['station_noise_seeds'][s.station_id],effective_snr_db=1) for i,s in enumerate(stations)])
    m={**p,'source_sha256':runner.source_hashes()};(tmp_path/'study_manifest.json').write_text(json.dumps(m))
    monkeypatch.setattr(runner,'load',lambda output:m)
    monkeypatch.setattr(runner,'source_bank',lambda *args:(np.ones(samples),dict(start_time_s=-2,signal_sha256='fixture')))
    monkeypatch.setattr(runner,'scenario',lambda *args:(stations,StationaryTrajectory([200,10,80]),np.zeros(3),dict(duration_s=samples/fs,time_origin_gazebo_s=0,state_sha256='fixture')))
    calls=[]
    def synth(*args,**kwargs):calls.append(kwargs);return stream
    monkeypatch.setattr(runner,'synthesize_multistation_audio',synth)
    runner.audio_worker(tmp_path,0);directory=runner.run_dir(tmp_path,0)
    runner.verify_completion(directory,runner.audio_context(tmp_path,0))
    rows=runner._measurement_rows(directory/'bearing_records.csv.gz')
    assert len(calls)==1 and len(rows)==36
    ids=[{r['event_id'] for r in rows if float(r['frontend_beta'])==b} for b in BETAS]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i))
    assert len({r['audio_stream_id'] for r in rows})==1
    (directory/'bearing_records.csv.gz').write_bytes(b'substituted')
    with pytest.raises(ValueError,match='SHA'):runner.verify_completion(directory,runner.audio_context(tmp_path,0))


def test_evaluation_requires_committed_selection_before_source_access(monkeypatch,tmp_path):
    from analysis import partial_whitening_study as runner
    monkeypatch.setattr(runner,'load',lambda output:runner.protocol())
    with pytest.raises(FileNotFoundError):runner.audio_worker(tmp_path,8)
    assert not (tmp_path/'runs').exists() and not (tmp_path/'source_banks').exists()


def test_freeze_rejects_evaluation_and_reinitialization(monkeypatch,tmp_path):
    from analysis import partial_whitening_report as report
    from analysis import partial_whitening_study as runner
    monkeypatch.setattr(report,'load',lambda output:runner.protocol())
    (tmp_path/'runs'/'08').mkdir(parents=True)
    with pytest.raises(ValueError,match='evaluation accessed'):report.freeze(tmp_path)
    with pytest.raises(FileExistsError):runner.initialize(tmp_path)


def test_external_budget_and_lock_stop_without_worker(monkeypatch,tmp_path):
    import json
    from analysis import partial_whitening_study as runner
    m=runner.protocol();m['limits']={**m['limits'],'total_worker_wall_s':1}
    monkeypatch.setattr(runner,'load',lambda output:m)
    (tmp_path/'execution_ledger.json').write_text(json.dumps(dict(attempts=[],worker_wall_s=0,stopped=False)))
    (tmp_path/'study_manifest.json').write_text('{}')
    monkeypatch.setattr(runner,'bounded_process',lambda *a,**k:pytest.fail('worker must not start'))
    (tmp_path/'worker.lock').write_text('another supervisor')
    with pytest.raises(FileExistsError):runner.execute(tmp_path,'development')
    (tmp_path/'worker.lock').unlink()
    runner.execute(tmp_path,'development')
    ledger=json.loads((tmp_path/'execution_ledger.json').read_text())
    assert ledger['stopped'] and ledger['stop_reason']=='budget_before_start' and not ledger['attempts']
    with pytest.raises(RuntimeError,match='technical stop'):runner.execute(tmp_path,'development')


@pytest.mark.parametrize('beta',BETAS)
def test_fft_nyquist_weight_and_noninteger_bounds(beta):
    x=np.random.default_rng(32).normal(size=32);y=np.roll(x,1)
    r=gcc_partial(x,y,FS,beta=beta,**{**KW,'minimum_frequency_hz':0,'maximum_frequency_hz':FS/2,'relative_spectral_floor':0})
    n,f,p,w=spectral_reference(x,y,beta,low=0,high=FS/2,floor=0)
    direct=np.real(np.sum(w[:,None]*p[:,None]*np.exp(2j*np.pi*f[:,None]*r.lags_samples[None,:]/FS),axis=0))/n
    np.testing.assert_allclose(r.correlation,direct,atol=2e-14,rtol=2e-13)
    assert r.lags_samples[0]==-12.3 and r.lags_samples[-1]==12.3


def test_new_experiment_init_and_empty_report(tmp_path):
    from analysis.partial_whitening_study import initialize,load
    from analysis.partial_whitening_report import report
    out=tmp_path/'new';m=initialize(out)
    assert load(out)==m
    assert m['planned_schedule']['0']['frames_per_station']>0
    r=report(out)
    assert r['audio_complete']==r['replay_computations']==0
    assert r['selected_beta'] is None and not r['technical_stop']
