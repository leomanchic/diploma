"""Read-only, frozen S8 harmonic/broadband failure diagnosis.

No signal synthesis or tracker invocation belongs to the analysis action.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, skew, kurtosis

from analysis.localization_error_attribution import _read_csv_gz, _write_csv_gz, _write_json
from analysis.robust_track_confirmation import METHODS, VARIANTS
from analysis.unseen_manoeuvres_and_sources import DEFAULT_OUTPUT as SOURCE, _directory, _load
from model.bearing_statistics import tangent_residual
from validation.gazebo_experiment import canonical_sha256, sha256, code_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'HARMONIC_TRACKING_FAILURE_PROTOCOL.md'
PROTOCOL_MANIFEST = ROOT / 'HARMONIC_TRACKING_FAILURE_PROTOCOL_MANIFEST.json'
DEFAULT_OUTPUT = ROOT / 'results/harmonic_tracking_failure'
SOURCE_RUNNER = 'analysis/unseen_manoeuvres_and_sources.py'


def verify_sources() -> dict:
    frozen = json.loads(PROTOCOL_MANIFEST.read_text(encoding='utf-8'))
    if sha256(PROTOCOL) != frozen['protocol_sha256']:
        raise ValueError('diagnostic protocol SHA mismatch')
    if code_sha256() != frozen['processing_code_sha256']:
        raise ValueError('original acoustic/tracker code SHA changed')
    for name, expected in frozen['source_files_sha256'].items():
        path = ROOT / (frozen['frozen_runner_path'] if name == SOURCE_RUNNER else name)
        if sha256(path) != expected:
            raise ValueError(f'historical input SHA mismatch: {name}')
    return frozen


def safe_corr(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def large_error_runs(times, errors, threshold_deg: float) -> list[dict]:
    """Exposure = n * grid step; invalid samples or missing epochs break runs."""
    t, e = np.asarray(times, float), np.asarray(errors, float)
    if len(t) < 2:
        return []
    if np.any(np.diff(t) <= 0):
        raise ValueError('diagnostic series must have increasing timestamps')
    step = float(np.median(np.diff(t)))
    result, start = [], None
    def emit(end):
        nonlocal start
        if start is not None:
            result.append({'start_reception_s': float(t[start]), 'last_reception_s': float(t[end]),
                           'frame_count': end-start+1, 'exposure_duration_s': (end-start+1)*step,
                           'threshold_deg': threshold_deg})
            start = None
    for i in range(len(t)):
        if i and t[i]-t[i-1] > step*1.01:
            emit(i-1)
        if np.isfinite(e[i]) and e[i] > threshold_deg:
            if start is None:
                start = i
        else:
            emit(i-1)
    emit(len(t)-1)
    return result


def calibration_diagnostics(residuals, bias, covariance) -> dict:
    e = np.asarray(residuals, float)
    b, r = np.asarray(bias, float), np.asarray(covariance, float)
    np.linalg.cholesky(r)
    if e.ndim != 2 or e.shape[1] != 2 or len(e) < 2:
        raise ValueError('at least two finite tangent residuals required')
    if not np.all(np.isfinite(e)):
        raise ValueError('nonfinite tangent residual')
    corrected = e-b
    d_before = np.einsum('ni,ni->n', e, np.linalg.solve(r, e.T).T)
    d_after = np.einsum('ni,ni->n', corrected, np.linalg.solve(r, corrected.T).T)
    empirical = np.cov(corrected, rowvar=False, ddof=1)
    white = np.linalg.solve(np.linalg.cholesky(r), corrected.T).T
    eigen = np.linalg.eigvalsh(np.cov(white, rowvar=False, ddof=1))
    return {
        'sample_count': len(e), 'mean_before_rad_json': json.dumps(np.mean(e, axis=0).tolist()),
        'mean_after_rad_json': json.dumps(np.mean(corrected, axis=0).tolist()),
        'bias_norm_deg': float(np.rad2deg(np.linalg.norm(b))),
        'empirical_mean_before_norm_deg': float(np.rad2deg(np.linalg.norm(np.mean(e,axis=0)))),
        'empirical_mean_after_norm_deg': float(np.rad2deg(np.linalg.norm(np.mean(corrected,axis=0)))),
        'empirical_covariance_rad2_json': json.dumps(empirical.tolist()),
        'whitened_covariance_eigenvalues_json': json.dumps(eigen.tolist()),
        'd_R2_median_before': float(np.median(d_before)), 'd_R2_median_after': float(np.median(d_after)),
        'd_R2_p95_after': float(np.percentile(d_after,95)),
        'd_R2_max_after': float(np.max(d_after)),
        'fraction_inside_nominal_chi2_95': float(np.mean(d_after <= chi2.ppf(.95,2))),
        'fraction_inside_fit_threshold': float(np.mean(d_after <= 10.596634733096073)),
        'fraction_d_R2_over_100': float(np.mean(d_after > 100)),
        'whitened_skew_json': json.dumps(skew(white,axis=0,bias=False).tolist()),
        'whitened_excess_kurtosis_json': json.dumps(kurtosis(white,axis=0,bias=False).tolist()),
        'evaluation_used_for_calibration': False,
    }


def _read_bearings(directory: Path, processing: dict) -> pd.DataFrame:
    frame = pd.read_csv(directory/'bearing_records.csv.gz',float_precision='round_trip',
                        dtype={'source_seed':str,'noise_seed':str,'sequence_id':str,'event_id':str})
    frame['tracker_selected'] = frame.frame_index % int(processing['tracker']['frame_stride']) == 0
    frame['error_deg_checked'] = np.nan
    frame['residual_0_rad_checked'] = np.nan
    frame['residual_1_rad_checked'] = np.nan
    truth = frame[['truth_local_'+a for a in 'xyz']].to_numpy(float)
    estimate = frame[['estimate_local_'+a for a in 'xyz']].to_numpy(float)
    valid = frame.valid.to_numpy(bool)
    e = np.array([tangent_residual(u,y) for u,y in zip(truth[valid],estimate[valid])])
    error = np.rad2deg(np.linalg.norm(e,axis=1))
    if not np.allclose(e, frame.loc[valid,['residual_az_arc_rad','residual_el_arc_rad']],rtol=1e-9,atol=1e-11):
        raise ValueError('saved residual differs from tracker prediction tangent convention')
    if not np.allclose(error,frame.loc[valid,'geodesic_error_deg'],rtol=1e-9,atol=1e-8):
        raise ValueError('saved angular error disagrees with independently reconstructed log map')
    frame.loc[valid,'error_deg_checked'] = error
    frame.loc[valid,['residual_0_rad_checked','residual_1_rad_checked']] = e
    return frame


def audit_pair(left: pd.DataFrame, right: pd.DataFrame, ls: dict, rs: dict,
               lm: dict, rm: dict) -> dict:
    for key in ('trajectory','distance_m','replicate','noise_seed','snr_ref_db'):
        if ls[key] != rs[key]:
            raise ValueError(f'pair {ls["index"]}/{rs["index"]} differs in {key}')
    for key in ('recording_state_sha256','standardized_noise_sha256','translation_enu_m','time_origin_gazebo_s'):
        if lm[key] != rm[key]:
            raise ValueError(f'paired bearing metadata differs in {key}')
    keys = ['station_id','estimator_variant','frame_index']
    left, right = left.sort_values(keys).reset_index(drop=True), right.sort_values(keys).reset_index(drop=True)
    exact = keys+['source_seed']  # source_seed intentionally excluded below
    for col in keys+['noise_seed','frame_start_reception_time_s','frame_center_reception_time_s',
                     'frame_end_reception_time_s','available_timestamp_s','true_emission_time_s_evaluator_only']:
        if not np.array_equal(left[col].to_numpy(),right[col].to_numpy()):
            raise ValueError(f'paired row schedule/noise differs in {col}')
    truth_columns = ['truth_'+kind+'_'+axis for kind in ('local','world') for axis in 'xyz']
    if not np.array_equal(left[truth_columns].to_numpy(),right[truth_columns].to_numpy()):
        raise ValueError('paired evaluator trajectory directions differ')
    return {'broadband_index':ls['index'],'harmonic_index':rs['index'],
            'distance_m':ls['distance_m'],'replicate':ls['replicate'],
            'row_count_each':len(left),'recording_state_sha256':lm['recording_state_sha256'],
            'standardized_noise_sha256':lm['standardized_noise_sha256'],
            'matching_trajectory_geometry_noise_schedule':True,
            'source_seed_intentionally_different':ls['source_seed']!=rs['source_seed'],
            'event_ids_disjoint':not bool(set(left.event_id)&set(right.event_id))}


def _base(spec: dict, method: str, **extra) -> dict:
    return {'index':spec['index'],'source_class':spec['source_class'],
            'distance_m':spec['distance_m'],'replicate':spec['replicate'],
            'method':method,**extra}


def _scores_details(raw, rows_by_id, threshold):
    scores = json.loads(raw or '[]')
    passed = [identity for identity,value in scores if np.isfinite(value) and value <= threshold]
    times = [float(rows_by_id[i]['frame_center_reception_time_s']) for i in passed]
    return {'count':len(scores),'passed_count':len(passed),
            'passed_station_count':len({rows_by_id[i]['station_id'] for i in passed}),
            'passed_reception_span_s':max(times)-min(times) if times else 0.,
            'finite_median':float(np.median([v for _,v in scores if np.isfinite(v)])) if any(np.isfinite(v) for _,v in scores) else None}


def chain_tables(spec, frame, directory, settings):
    summaries, fits_out, hypotheses_out, generations_out, timeline = [], [], [], [], []
    fit_threshold = settings['tracker']['recovery']['fit_nis_threshold']
    for method in METHODS:
        selected = frame[(frame.estimator_variant==method)&frame.tracker_selected]
        by_id = {row['event_id']:row for row in selected.to_dict('records')}
        epochs = sorted(set(selected.available_timestamp_s))
        for variant in VARIANTS:
            suffix = f'{method}_{variant}.csv.gz'
            logs = {kind:_read_csv_gz(directory/f'{kind}_{suffix}') for kind in
                    ('tracking','updates','batch_fits','hypotheses','event_uses','lifecycle')}
            tracks, fits, hypotheses = logs['tracking'],logs['batch_fits'],logs['hypotheses']
            if not np.array_equal(epochs,[float(row['processing_time_s']) for row in tracks]):
                raise ValueError(f'incomplete saved publication schedule: {spec["index"]}/{suffix}')
            base = _base(spec,method,confirmation_variant=variant)
            count_by_generation=Counter()
            for f in fits:
                ids=json.loads(f['event_ids']); observations=[by_id[i] for i in ids]
                t=float(f['processing_time_s']); g=int(f['generation'])
                if f['reason']!='computational_budget_exceeded_before_fit':count_by_generation[g]+=1
                available=selected[(selected.available_timestamp_s<=t)&selected.valid]
                recv=[x['frame_center_reception_time_s'] for x in observations]
                fits_out.append({**base,**f,'available_valid_event_count':len(available),
                    'available_station_count':available.station_id.nunique(),
                    'fit_event_count':len(ids),'fit_station_count':len({x['station_id'] for x in observations}),
                    'fit_reception_span_s':max(recv)-min(recv) if recv else 0.})
            for g,count in sorted(count_by_generation.items()):
                generations_out.append({**base,'generation':g,'executed_optimizations':count})
            for h in hypotheses:
                detail={**base,**h}
                for label,key in (('preliminary_R','preliminary_nis_values'),('predictive_S','confirmation_nis_values'),('final_R','final_nis_values')):
                    detail.update({label+'_'+k:v for k,v in _scores_details(h[key],by_id,fit_threshold).items()})
                hypotheses_out.append(detail)
            valid=[t for t in tracks if t['valid']=='True']
            created=[h for h in hypotheses if h['action']=='tentative_created']
            summaries.append({**base,'available_event_count':len(selected),
                'valid_input_event_count':int(selected.valid.sum()),'input_station_count':selected.station_id.nunique(),
                'input_reception_span_s':float(selected.frame_center_reception_time_s.max()-selected.frame_center_reception_time_s.min()),
                'publication_count':len(tracks),'valid_publication_count':len(valid),
                'first_confirmation_s':float(valid[0]['processing_time_s']) if valid else None,
                'first_position_error_m':float(valid[0]['position_error_m']) if valid else None,
                'created_hypothesis_count':len(created),'executed_optimizations':sum(count_by_generation.values()),
                'optimizations_by_generation_json':json.dumps(dict(count_by_generation),sort_keys=True),
                'fit_reasons_json':json.dumps(dict(Counter(f['reason'] for f in fits)),sort_keys=True),
                'batch_failure_reasons_json':json.dumps(dict(Counter(f['batch_failure_reason'] for f in fits if f['batch_failure_reason'])),sort_keys=True),
                'hypothesis_actions_reasons_json':json.dumps(dict(Counter(h['action']+':'+h['reason'] for h in hypotheses)),sort_keys=True),
                'publication_failure_reasons_json':json.dumps(dict(Counter(t['failure_reason'] for t in tracks if t['failure_reason'])),sort_keys=True),
                'initialization_used_event_count':sum(e['role']=='initialization' for e in logs['event_uses']),
                'accepted_update_count':sum(u['update_applied']=='True' for u in logs['updates']),
                'rejected_update_count':sum(u['update_applied']!='True' for u in logs['updates']),
                'hypothesis_limit_observed':any('hypothesis_limit' in t['failure_reason'] for t in tracks),
                'external_timeout':False,'candidate_geometric_rejections_saved':False})
            if spec['index'] in (0,6):
                for track in tracks:
                    t=float(track['processing_time_s']);available=selected[selected.available_timestamp_s<=t]
                    prior_fits=[f for f in fits if float(f['processing_time_s'])<=t]
                    at_h=[h for h in hypotheses if float(h['processing_time_s'])==t]
                    prior_created=[h for h in created if float(h['processing_time_s'])<=t]
                    timeline.append({**base,'processing_time_s':t,'available_event_count':len(available),
                        'available_station_count':available.station_id.nunique(),
                        'available_reception_span_s':float(available.frame_center_reception_time_s.max()-available.frame_center_reception_time_s.min()),
                        'executed_optimizations_cumulative':sum(f['reason']!='computational_budget_exceeded_before_fit' for f in prior_fits),
                        'hypotheses_created_cumulative':len(prior_created),
                        'hypothesis_actions_at_time_json':json.dumps([(h['action'],h['reason']) for h in at_h]),
                        'status':track['status'],'valid':track['valid']=='True','failure_reason':track['failure_reason'],
                        'position_error_m':float(track['position_error_m']) if track['position_error_m'] else None})
    return summaries,fits_out,hypotheses_out,generations_out,timeline


def analyze(output: Path=DEFAULT_OUTPUT) -> dict:
    frozen=verify_sources(); manifest=_load(SOURCE)
    output=Path(output)
    if output.exists():
        raise FileExistsError('new derivative output required')
    output.mkdir(parents=True)
    frames, metadata = {},{}
    bearing,calibration,runs,temporal,interstation,residual_rows=[],[],[],[],[],[]
    chain,fit_rows,hypothesis_rows,generation_rows,timeline=[],[],[],[],[]
    calibration_map={(v['station_id'],v['estimator_variant']):v for v in manifest['processing']['calibration']['values']}
    for index in sorted({i for pair in frozen['pairs'] for i in pair}):
        spec=manifest['specs'][index];directory=_directory(SOURCE,spec)
        experiment=json.loads((directory/'experiment.json').read_text())
        if experiment['status']!='complete':raise ValueError('preselected input is incomplete')
        for name,digest in experiment['result_sha256'].items():
            if sha256(directory/name)!=digest:raise ValueError(f'changed input {index}/{name}')
        frame=_read_bearings(directory,manifest['processing']);frames[index]=frame
        metadata[index]=json.loads((directory/'bearing_manifest.json').read_text())
        for (station,method),group in frame.groupby(['station_id','estimator_variant']):
            config=calibration_map[station,method]
            for subset in ('all_frames','tracker_selected'):
                selected=group if subset=='all_frames' else group[group.tracker_selected]
                base=_base(spec,method,station_id=station,subset=subset)
                valid=selected[selected.valid];errors=valid.error_deg_checked.to_numpy()
                values={'total_frames':len(selected),'valid_frames':len(valid),'valid_fraction':len(valid)/len(selected),
                        'median_error_deg':float(np.median(errors)) if len(valid) else None,
                        'p95_error_deg':float(np.percentile(errors,95)) if len(valid) else None,
                        'maximum_error_deg':float(np.max(errors)) if len(valid) else None}
                for threshold in (5.,10.,30.):
                    series=large_error_runs(selected.frame_center_reception_time_s,selected.error_deg_checked,threshold)
                    values[f'fraction_valid_over_{int(threshold)}deg']=float(np.mean(errors>threshold)) if len(valid) else None
                    values[f'largest_run_over_{int(threshold)}deg_s']=max((r['exposure_duration_s'] for r in series),default=0.)
                    runs.extend({**base,**r} for r in series)
                bearing.append({**base,**values})
                residual=valid[['residual_0_rad_checked','residual_1_rad_checked']].to_numpy()
                calibration.append({**base,**calibration_diagnostics(residual,config['bias_rad'],config['covariance_rad2'])})
                step=float(np.median(np.diff(selected.frame_center_reception_time_s)))
                for lag in sorted({1,max(1,round(.5/step)),max(1,round(1./step))}):
                    for col in ('error_deg_checked','residual_0_rad_checked','residual_1_rad_checked'):
                        arr=selected[col].to_numpy()
                        temporal.append({**base,'component':col,'lag_frames':lag,'lag_s':lag*step,
                                         'correlation':safe_corr(arr[:-lag],arr[lag:])})
            valid=group[group.valid]
            e=valid[['residual_0_rad_checked','residual_1_rad_checked']].to_numpy()
            ec=e-np.asarray(config['bias_rad'])
            d=np.einsum('ni,ni->n',ec,np.linalg.solve(np.asarray(config['covariance_rad2']),ec.T).T)
            for row,raw,corrected,dr in zip(valid.to_dict('records'),e,ec,d):
                residual_rows.append({**_base(spec,method,station_id=station),'event_id':row['event_id'],
                    'frame_index':row['frame_index'],'reception_time_s':row['frame_center_reception_time_s'],
                    'availability_time_s':row['available_timestamp_s'],'tracker_selected':row['tracker_selected'],
                    'angular_error_deg':row['error_deg_checked'],'raw_0_rad':raw[0],'raw_1_rad':raw[1],
                    'corrected_0_rad':corrected[0],'corrected_1_rad':corrected[1],'d_R2':dr})
        for method in METHODS:
            for subset in ('all_frames','tracker_selected'):
                selected=frame[frame.estimator_variant==method]
                if subset=='tracker_selected':selected=selected[selected.tracker_selected]
                piv=selected.pivot(index='frame_index',columns='station_id',values='error_deg_checked')
                stations=list(piv.columns)
                for a in range(len(stations)):
                    for b in range(a+1,len(stations)):
                        x,y=piv.iloc[:,a].to_numpy(),piv.iloc[:,b].to_numpy();mask=np.isfinite(x)&np.isfinite(y)
                        both=(x[mask]>10)&(y[mask]>10);either=(x[mask]>10)|(y[mask]>10)
                        interstation.append({**_base(spec,method,subset=subset),'station_a':stations[a],
                            'station_b':stations[b],'paired_valid_frames':int(mask.sum()),
                            'angular_error_correlation':safe_corr(x,y),
                            'joint_over_10deg_fraction':float(np.mean(both)) if mask.any() else None,
                            'jaccard_over_10deg':float(both.sum()/either.sum()) if either.any() else None})
        parts=chain_tables(spec,frame,directory,manifest['processing'])
        for destination,rows in zip((chain,fit_rows,hypothesis_rows,generation_rows,timeline),parts):destination.extend(rows)
        print(f'analyzed {index:02d}',flush=True)
    pairs=[audit_pair(frames[l],frames[r],manifest['specs'][l],manifest['specs'][r],metadata[l],metadata[r]) for l,r in frozen['pairs']]
    tables={'bearing_metrics':bearing,'calibration_metrics':calibration,'large_error_runs':runs,
            'temporal_correlations':temporal,'interstation_correlations':interstation,
            'bearing_residuals':residual_rows,'chain_summary':chain,'batch_attempts':fit_rows,
            'hypothesis_details':hypothesis_rows,'fit_generation_counts':generation_rows,'failure_timeline_200m':timeline}
    files={}
    for name,rows in tables.items():
        path=output/(name+'.csv.gz')
        _write_csv_gz(path,rows,columns=list(rows[0]) if rows else ('index',))
        files[path.name]=sha256(path)
    _write_json(output/'pair_audit.json',pairs);files['pair_audit.json']=sha256(output/'pair_audit.json')
    result={'schema_version':1,'status':'read_only_analysis_complete','protocol_manifest_sha256':sha256(PROTOCOL_MANIFEST),
            'protocol_freeze_commit':'3bbf527','source_manifest_sha256':sha256(SOURCE/'evaluation_manifest.json'),
            'analyzer_sha256':sha256(Path(__file__)),'processing_code_sha256':code_sha256(),
            'sequence_count':10,'paired_sequence_count':5,'original_tracker_count':40,
            'replays_executed_by_analysis':0,'rows':{k:len(v) for k,v in tables.items()},'files_sha256':files,
            'missing_evidence':['pair correlation curves and selected lag peaks not saved',
                'geometric candidate rejection journal not exported by historical runner',
                'hypothesis states/covariances and every interim confirmation decision not exported']}
    _write_json(output/'analysis_manifest.json',result)
    verify_sources()
    return result


def verify(output=DEFAULT_OUTPUT):
    verify_sources();output=Path(output)
    result=json.loads((output/'analysis_manifest.json').read_text())
    if result['analyzer_sha256']!=sha256(Path(__file__)):raise ValueError('analyzer SHA changed')
    if result['protocol_manifest_sha256']!=sha256(PROTOCOL_MANIFEST):raise ValueError('protocol manifest changed')
    for name,digest in result['files_sha256'].items():
        if sha256(output/name)!=digest:raise ValueError(f'derivative SHA mismatch: {name}')
    return {'status':'verified','sequence_count':result['sequence_count'],'historical_inputs_unchanged':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('analyze','verify'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=analyze(args.output) if args.action=='analyze' else verify(args.output)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
