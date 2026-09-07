"""Space-network event count filtered by AMIGO follow-up SNR."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import os

import numpy as np
import scipy.integrate
import scipy.interpolate

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent / '.matplotlib-cache'))

module_path = Path(__file__).with_name('test_notebook_functions.py')
spec = importlib.util.spec_from_file_location('tnf', module_path)
tnf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tnf)


R0 = 19.0
T_OBS_LISA_YR = 4.0
T_OBS_AMIGO_YR = 4.0
T_DELAY_YR = 0.0
T_WINDOW_YR = [max(0, T_DELAY_YR), T_OBS_LISA_YR + T_OBS_AMIGO_YR + T_DELAY_YR]
RHO_THR = 8.0
Z_MAX = 0.3
T_MERGER_GRID_SIZE = 100
AMIGO_F_MIN = 1.0e-2
AMIGO_F_MAX = 10.0
AMIGO_N_FREQ = 2000
LAMIGO_VALUES = (1.0e7, 5.0e7)

N_MC_TOTAL = 1000
N_WORKERS = os.cpu_count() or 4
SEED_BASE = 12345
Q_MIN = 0.1
Q_MAX = 1.0
MASS_GRID_SIZE = 10000
REDSHIFT_GRID_SIZE = 10000

DETECTOR_ALIASES = {
    'lisa': 'LISA',
    'taiji': 'Taiji',
    'tianqin': 'TianQin',
    'tian-qin': 'TianQin',
    'tian_qin': 'TianQin',
}

RUN_CONFIG = {
    'detector': 'LISA-Taiji-TianQin',
    'R0': R0,
    'LAMIGO': LAMIGO_VALUES,
    'Tobs': T_OBS_LISA_YR,
    'Tdelay': T_DELAY_YR,
    'rho_thr': RHO_THR,
    'z_max': Z_MAX,
    'n_mc': N_MC_TOTAL,
    'n_workers': N_WORKERS,
    'seed': SEED_BASE,
    'out_samples': None,
    'out_summary': None,
}

def parse_detector_network(detector):
    normalized = str(detector).replace(',', '-').replace('Tian-Qin', 'TianQin').replace('tian-qin', 'tianqin')
    parts = [part.strip() for part in normalized.split('-') if part.strip()]
    detectors = []
    for part in parts:
        key = part.lower()
        if key not in DETECTOR_ALIASES:
            valid = ', '.join(sorted(set(DETECTOR_ALIASES.values())))
            raise ValueError(f"Unknown detector {part!r}; choose a '-' separated network from {valid}")
        canonical = DETECTOR_ALIASES[key]
        if canonical not in detectors:
            detectors.append(canonical)
    if not detectors:
        raise ValueError('At least one detector is required')
    return tuple(detectors)


def detector_tag(detectors):
    return '-'.join(detectors).replace('TianQin', 'TianQin')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Parallel single-case LISA/Taiji/TianQin space-network event count filtered by AMIGO follow-up SNR, FullPop masses, and Madau redshifts'
    )
    parser.add_argument('--detector', type=str, default=RUN_CONFIG['detector'],
                        help='Space detector network, e.g. LISA, Taiji, TianQin, LISA-Taiji, LISA-TianQin, Taiji-TianQin, or LISA-Taiji-TianQin.')
    parser.add_argument('--R0', type=float, default=RUN_CONFIG['R0'])
    parser.add_argument(
        '--LAMIGO',
        type=float,
        nargs='+',
        default=list(RUN_CONFIG['LAMIGO']),
        help='One or more AMIGO arm lengths in meters (default: 1e7 5e7).',
    )
    parser.add_argument('--Tobs', type=float, default=RUN_CONFIG['Tobs'])
    parser.add_argument(
        '--Tdelay',
        type=float,
        nargs='+',
        default=[RUN_CONFIG['Tdelay']],
        help='One or more delay times in years, e.g. --Tdelay 0 1 2.5',
    )
    parser.add_argument('--rho-thr', type=float, default=RUN_CONFIG['rho_thr'])
    parser.add_argument('--z-max', type=float, default=RUN_CONFIG['z_max'])
    parser.add_argument('--n-mc', type=int, default=RUN_CONFIG['n_mc'])
    parser.add_argument('--n-workers', type=int, default=RUN_CONFIG['n_workers'])
    parser.add_argument('--seed', type=int, default=RUN_CONFIG['seed'])
    parser.add_argument('--out-samples', type=Path, default=RUN_CONFIG['out_samples'])
    parser.add_argument('--out-summary', type=Path, default=RUN_CONFIG['out_summary'])
    args, _ = parser.parse_known_args(argv)
    return args


def args_from_run_config():
    return argparse.Namespace(
        detector=RUN_CONFIG['detector'],
        R0=RUN_CONFIG['R0'],
        LAMIGO=list(RUN_CONFIG['LAMIGO']),
        Tobs=RUN_CONFIG['Tobs'],
        Tdelay=[RUN_CONFIG['Tdelay']],
        rho_thr=RUN_CONFIG['rho_thr'],
        z_max=RUN_CONFIG['z_max'],
        n_mc=RUN_CONFIG['n_mc'],
        n_workers=RUN_CONFIG['n_workers'],
        seed=RUN_CONFIG['seed'],
        out_samples=RUN_CONFIG['out_samples'],
        out_summary=RUN_CONFIG['out_summary'],
    )


def case_tag(detectors, r0, t_obs_yr, t_delay_yr, rho_thr, z_max):
    r0_tag = str(r0).replace('.', 'p')
    tobs_tag = str(t_obs_yr).replace('.', 'p')
    tdelay_tag = str(t_delay_yr).replace('.', 'p')
    rho_tag = str(rho_thr).replace('.', 'p')
    z_tag = str(z_max).replace('.', 'p')
    det_tag = detector_tag(detectors).replace('-', '_').lower()
    return f'{det_tag}_fullpop_madau_R0-{r0_tag}_Tobs-{tobs_tag}_Tdelay-{tdelay_tag}_rho-{rho_tag}_zmax-{z_tag}'


def case_tag_multi(detectors, r0, t_obs_yr, t_delay_values, rho_thr, z_max):
    if len(t_delay_values) == 1:
        return case_tag(detectors, r0, t_obs_yr, t_delay_values[0], rho_thr, z_max)
    r0_tag = str(r0).replace('.', 'p')
    tobs_tag = str(t_obs_yr).replace('.', 'p')
    rho_tag = str(rho_thr).replace('.', 'p')
    z_tag = str(z_max).replace('.', 'p')
    delay_min = str(min(t_delay_values)).replace('.', 'p')
    delay_max = str(max(t_delay_values)).replace('.', 'p')
    det_tag = detector_tag(detectors).replace('-', '_').lower()
    return f'{det_tag}_fullpop_madau_R0-{r0_tag}_Tobs-{tobs_tag}_Tdelay-list-{len(t_delay_values)}_{delay_min}-to-{delay_max}_rho-{rho_tag}_zmax-{z_tag}'


def t_delay_values_from_args(args):
    values = args.Tdelay if isinstance(args.Tdelay, (list, tuple)) else [args.Tdelay]
    return [float(value) for value in values]


def lamigo_values_from_args(args):
    values = args.LAMIGO if isinstance(args.LAMIGO, (list, tuple)) else [args.LAMIGO]
    return [float(value) for value in values]


def t_window_yr(t_obs_lisa_yr, t_delay_yr):
    return max(0.0, t_delay_yr), t_obs_lisa_yr + T_OBS_AMIGO_YR + t_delay_yr


def t_merger_grid_yr(t_obs_lisa_yr, t_delay_yr):
    window = t_window_yr(t_obs_lisa_yr, t_delay_yr)
    return np.linspace(window[0], window[1], T_MERGER_GRID_SIZE)

def sample_from_pdf_grid(grid, pdf, size, rng):
    cdf = scipy.integrate.cumulative_trapezoid(pdf, grid, initial=0.0)
    if cdf[-1] <= 0.0:
        raise ValueError('PDF grid has non-positive normalization')
    cdf /= cdf[-1]
    inverse_cdf = scipy.interpolate.interp1d(
        cdf,
        grid,
        bounds_error=False,
        fill_value=(grid[0], grid[-1]),
        assume_sorted=True,
    )
    return inverse_cdf(rng.random(size))


def mass_ratio_pdf(q, beta_q, q_min=Q_MIN, q_max=Q_MAX):
    q = np.asarray(q, dtype=float)
    pdf = np.where((q >= q_min) & (q <= q_max), q ** beta_q, 0.0)
    norm_grid = np.linspace(q_min, q_max, 2000)
    norm = scipy.integrate.trapezoid(norm_grid ** beta_q, norm_grid)
    return pdf / norm


def sample_mass_ratio(size, beta_q, rng, q_min=Q_MIN, q_max=Q_MAX):
    u = rng.random(size)
    power = beta_q + 1.0
    if abs(power) > 1.0e-12:
        return (u * (q_max**power - q_min**power) + q_min**power) ** (1.0 / power)
    return q_min * (q_max / q_min) ** u


def sample_fullpop_masses(size, rng):
    m_min = tnf.injected['m_min']
    m_max = tnf.injected['m_max']
    m1_grid = np.geomspace(m_min + 0.001, m_max, MASS_GRID_SIZE)
    p_m1_grid = tnf.fullpop40_primary_mass_pdf(m1_grid, tnf.injected)
    m1 = sample_from_pdf_grid(m1_grid, p_m1_grid, size, rng)
    q = sample_mass_ratio(size, tnf.injected['beta_q'], rng)
    m2 = np.maximum(m_min, q * m1)
    m2 = np.minimum(m2, m1)
    q = m2 / m1
    return m1, m2, q


def madau_redshift_weight_grid(z_max):
    z_grid = np.linspace(1.0e-4, z_max, REDSHIFT_GRID_SIZE)
    dc_grid = np.array([tnf.comoving_distance_mpc(z) for z in z_grid])
    dvc_dz = 4.0 * np.pi * ((tnf.c / 1000.0) / tnf.H0) * dc_grid**2 / tnf.e_z(z_grid) / 1.0e9
    rate_shape = tnf.madau_dickinson_rate(
        z_grid,
        gamma=tnf.injected['gamma'],
        kappa=tnf.injected['kappa'],
        zp=tnf.injected['z_p'],
        R0=1.0,
    )
    weight = rate_shape * dvc_dz / (1.0 + z_grid)
    return z_grid, weight


def sample_madau_redshift(size, rng, z_max):
    z_grid, weight = madau_redshift_weight_grid(z_max)
    return sample_from_pdf_grid(z_grid, weight, size, rng)


def redshift_weight_integral(z_max):
    z_grid, weight = madau_redshift_weight_grid(z_max)
    return float(scipy.integrate.trapezoid(weight, z_grid))

def space_network_snr_vs_merger_time(m1, m2, luminosity_distance_mpc, merger_times_yr, detectors, observation_years):
    rho_squared = np.zeros_like(merger_times_yr, dtype=float)
    detector_curves = {}
    redshift = None
    for detector in detectors:
        z, rho_curve = tnf.snr_vs_merger_time(
            m1,
            m2,
            luminosity_distance_mpc,
            merger_times_yr,
            detector=detector,
            observation_years=observation_years,
        )
        redshift = z
        detector_curves[detector] = rho_curve
        rho_squared += rho_curve**2
    return redshift, np.sqrt(rho_squared), detector_curves


def threshold_segments(x, y, threshold):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    above = y >= threshold
    segments = []
    start = x[0] if above[0] else None
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if above[i] == above[i + 1]:
            continue
        if y2 == y1:
            crossing = x[i]
        else:
            frac = (threshold - y1) / (y2 - y1)
            crossing = x[i] + frac * (x[i + 1] - x[i])
        if above[i] and not above[i + 1]:
            segments.append((float(start), float(crossing)))
            start = None
        elif not above[i] and above[i + 1]:
            start = float(crossing)
    if above[-1] and start is not None:
        segments.append((float(start), float(x[-1])))
    return [(lo, hi) for lo, hi in segments if hi > lo]


def segments_to_bounds(segments):
    if not segments:
        return 0.0, np.nan, np.nan
    total = float(sum(hi - lo for lo, hi in segments))
    lower = float(min(lo for lo, _ in segments))
    upper = float(max(hi for _, hi in segments))
    return total, lower, upper


def space_time_window_bounds(m1, m2, z, observation_years, t_delay_yr, rho_thr, detectors):
    dl_mpc = float(tnf.luminosity_distance_mpc_from_z(float(z)))
    merger_grid_yr = t_merger_grid_yr(observation_years, t_delay_yr)
    _, space_rho_curve, _ = space_network_snr_vs_merger_time(
        m1,
        m2,
        dl_mpc,
        merger_grid_yr,
        detectors,
        observation_years,
    )
    space_segments = threshold_segments(merger_grid_yr, space_rho_curve, rho_thr)
    return segments_to_bounds(space_segments)


def amigo_full_band_snr(m1, m2, z, lamigo):
    dl_mpc = float(tnf.luminosity_distance_mpc_from_z(float(z)))
    frequencies = np.geomspace(AMIGO_F_MIN, AMIGO_F_MAX, AMIGO_N_FREQ)
    h_tilde = tnf.phenoma_fourier_amplitude(
        frequencies,
        m1,
        m2,
        z,
        dl_mpc * 1.0e6 * tnf.pc,
    )
    integrand = h_tilde**2 / tnf.amigo_psd(frequencies, lamigo)
    snr = np.sqrt((16.0 / 5.0) * scipy.integrate.trapezoid(integrand, frequencies))
    return float(snr)

def worker_chunk(args):
    detectors, t_obs_yr, t_delay_yr, rho_thr, z_max, lamigo_values, n_samples, seed = args
    rng = np.random.default_rng(seed)
    m1, m2, q = sample_fullpop_masses(n_samples, rng)
    z = sample_madau_redshift(n_samples, rng, z_max)
    space_windows = np.zeros(n_samples)
    t_merger_min = np.full(n_samples, np.nan)
    t_merger_max = np.full(n_samples, np.nan)
    amigo_snr = np.full((len(lamigo_values), n_samples), np.nan)
    amigo_detectable = np.zeros((len(lamigo_values), n_samples), dtype=bool)

    for i in range(n_samples):
        window, lower, upper = space_time_window_bounds(m1[i], m2[i], z[i], t_obs_yr, t_delay_yr, rho_thr, detectors)
        space_windows[i] = window
        t_merger_min[i] = lower
        t_merger_max[i] = upper
        if window > 0.0:
            for lamigo_index, lamigo in enumerate(lamigo_values):
                amigo_snr[lamigo_index, i] = amigo_full_band_snr(m1[i], m2[i], z[i], lamigo)
                amigo_detectable[lamigo_index, i] = amigo_snr[lamigo_index, i] >= rho_thr

    return {
        'count': n_samples,
        'm1': m1,
        'm2': m2,
        'q': q,
        'z': z,
        'space_windows': space_windows,
        't_merger_min': t_merger_min,
        't_merger_max': t_merger_max,
        'amigo_snr': amigo_snr,
        'amigo_detectable': amigo_detectable,
    }


def run_case(detectors, r0, t_obs_yr, t_delay_yr, rho_thr, z_max, lamigo_values, n_mc, n_workers, seed_base):
    detectors = parse_detector_network('-'.join(detectors) if isinstance(detectors, (tuple, list)) else detectors)
    lamigo_values = tuple(float(value) for value in lamigo_values)
    n_workers = max(1, min(int(n_workers), int(n_mc)))
    base = n_mc // n_workers
    remainder = n_mc % n_workers
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    cases = [
        (detectors, t_obs_yr, t_delay_yr, rho_thr, z_max, lamigo_values, chunk_sizes[i], seed_base + i)
        for i in range(n_workers)
        if chunk_sizes[i] > 0
    ]

    if len(cases) == 1:
        results = [worker_chunk(cases[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(worker_chunk, cases))

    total_count = sum(r['count'] for r in results)
    arrays = {
        name: np.concatenate([r[name] for r in results])
        for name in [
            'm1',
            'm2',
            'q',
            'z',
            'space_windows',
            't_merger_min',
            't_merger_max',
        ]
    }
    arrays['amigo_snr'] = np.concatenate([r['amigo_snr'] for r in results], axis=1)
    arrays['amigo_detectable'] = np.concatenate([r['amigo_detectable'] for r in results], axis=1)

    total_weight = redshift_weight_integral(z_max)
    mean_window = float(np.mean(arrays['space_windows']))
    nspace = r0 * total_weight * mean_window
    rspace = nspace / t_obs_yr
    space_detectable = arrays['space_windows'] > 0.0
    window = t_window_yr(t_obs_yr, t_delay_yr)

    amigo_results = {}
    for lamigo_index, lamigo in enumerate(lamigo_values):
        space_amigo_detectable = space_detectable & arrays['amigo_detectable'][lamigo_index]
        amigo_filtered_windows = np.where(space_amigo_detectable, arrays['space_windows'], 0.0)
        mean_amigo_filtered_window = float(np.mean(amigo_filtered_windows))
        nspace_amigo = r0 * total_weight * mean_amigo_filtered_window
        amigo_results[lamigo] = {
            'n_space_amigo_detected': int(np.count_nonzero(space_amigo_detectable)),
            'mean_space_amigo_filtered_window_yr': mean_amigo_filtered_window,
            'Nspace_amigo': nspace_amigo,
            'rspace_amigo': nspace_amigo / t_obs_yr,
        }

    summary = {
        'model': 'fullpop_madau',
        'detection_mode': 'space_network_detected_then_multi_LAMIGO_threshold_filtered',
        'space_detector_network': detector_tag(detectors),
        'space_detectors': detectors,
        'followup_detector': 'AMIGO',
        'LAMIGO_values_m': list(lamigo_values),
        'R0': r0,
        'Tobs_space_yr': t_obs_yr,
        'Tobs_amigo_yr': T_OBS_AMIGO_YR,
        'Tdelay_yr': t_delay_yr,
        'T_window_yr': window,
        'AMIGO_f_min_Hz': AMIGO_F_MIN,
        'AMIGO_f_max_Hz': AMIGO_F_MAX,
        'AMIGO_n_freq': AMIGO_N_FREQ,
        'rho_thr': rho_thr,
        'z_max': z_max,
        'n_samples': total_count,
        'n_workers': n_workers,
        'n_space_window_positive': int(np.count_nonzero(space_detectable)),
        'redshift_weight_integral_Gpc3': total_weight,
        'mean_space_window_yr': mean_window,
        'Nspace': nspace,
        'rspace': rspace,
        'AMIGO_results': amigo_results,
        'Nspace_amigo_by_LAMIGO': {
            lamigo: result['Nspace_amigo'] for lamigo, result in amigo_results.items()
        },
    }
    return summary, arrays

def save_detectable_samples(path, arrays, summary):
    path = Path(path)
    space_detectable = arrays['space_windows'] > 0.0
    m1 = arrays['m1'][space_detectable]
    m2 = arrays['m2'][space_detectable]
    q = arrays['q'][space_detectable]
    z = arrays['z'][space_detectable]
    windows = arrays['space_windows'][space_detectable]
    t_min = arrays['t_merger_min'][space_detectable]
    t_max = arrays['t_merger_max'][space_detectable]
    chirp_mass = tnf.chirp_mass_source(m1, m2)

    summary['n_saved_space_window_positive'] = int(len(m1))
    columns = [
        m1,
        m2,
        chirp_mass,
        q,
        z,
        windows,
        t_min,
        t_max,
    ]
    column_names = [
        'm1_msun',
        'm2_msun',
        'chirp_mass_msun',
        'q',
        'redshift',
        'space_window_yr',
        't_merger_min_yr',
        't_merger_max_yr',
    ]
    for lamigo_index, lamigo in enumerate(summary['LAMIGO_values_m']):
        columns.extend([
            arrays['amigo_snr'][lamigo_index, space_detectable],
            arrays['amigo_detectable'][lamigo_index, space_detectable].astype(int),
        ])
        tag = f'{lamigo:.0e}'.replace('+', '')
        column_names.extend([f'amigo_snr_LAMIGO_{tag}', f'amigo_detectable_LAMIGO_{tag}'])

    samples = np.column_stack(columns)
    header_lines = [
        f"{summary['space_detector_network']}-detected FullPop+Madau samples with AMIGO SNR evaluated for every LAMIGO",
        f'summary = {summary}',
        f"columns: {' '.join(column_names)}",
    ]
    np.savetxt(path, samples, header='\n'.join(header_lines), fmt='%.10e')


def save_detectable_samples_multi(path, result_entries, aggregate_summary):
    rows = []
    for entry in result_entries:
        arrays = entry['arrays']
        summary = entry['summary']
        space_detectable = arrays['space_windows'] > 0.0
        if not np.any(space_detectable):
            continue
        m1 = arrays['m1'][space_detectable]
        m2 = arrays['m2'][space_detectable]
        q = arrays['q'][space_detectable]
        z = arrays['z'][space_detectable]
        windows = arrays['space_windows'][space_detectable]
        t_min = arrays['t_merger_min'][space_detectable]
        t_max = arrays['t_merger_max'][space_detectable]
        chirp_mass = tnf.chirp_mass_source(m1, m2)
        t_delay = np.full_like(m1, summary['Tdelay_yr'], dtype=float)
        columns = [
            t_delay,
            m1,
            m2,
            chirp_mass,
            q,
            z,
            windows,
            t_min,
            t_max,
        ]
        for lamigo_index in range(len(summary['LAMIGO_values_m'])):
            columns.extend([
                arrays['amigo_snr'][lamigo_index, space_detectable],
                arrays['amigo_detectable'][lamigo_index, space_detectable].astype(int),
            ])
        rows.append(np.column_stack(columns))
        summary['n_saved_space_window_positive'] = int(len(m1))

    column_names = [
        'Tdelay_yr',
        'm1_msun',
        'm2_msun',
        'chirp_mass_msun',
        'q',
        'redshift',
        'space_window_yr',
        't_merger_min_yr',
        't_merger_max_yr',
    ]
    for lamigo in aggregate_summary['LAMIGO_values_m']:
        tag = f'{lamigo:.0e}'.replace('+', '')
        column_names.extend([f'amigo_snr_LAMIGO_{tag}', f'amigo_detectable_LAMIGO_{tag}'])
    samples = np.vstack(rows) if rows else np.empty((0, len(column_names)))
    header_lines = [
        f"{aggregate_summary['space_detector_network']}-detected FullPop+Madau samples with all AMIGO LAMIGO results on each row",
        f'aggregate_summary = {aggregate_summary}',
        f"columns: {' '.join(column_names)}",
    ]
    np.savetxt(path, samples, header='\n'.join(header_lines), fmt='%.10e')


def output_paths(args, detectors, t_delay_values, lamigo_values):
    tag = case_tag_multi(detectors, args.R0, args.Tobs, t_delay_values, args.rho_thr, args.z_max)
    lamigo_tag = '-'.join(f'{value:.0e}'.replace('+', '') for value in lamigo_values)
    tag = f'{tag}_LAMIGO-{lamigo_tag}'
    output_dir = module_path.parent
    out_summary = args.out_summary or output_dir / f'space_event_count_amigo_fullband_detected_{tag}.txt'
    out_samples = args.out_samples or output_dir / f'space_event_count_amigo_fullband_detected_{tag}_samples.txt'
    return Path(out_summary), Path(out_samples)


def execute(args=None):
    if args is None:
        args = args_from_run_config()
    detectors = parse_detector_network(args.detector)
    t_delay_values = t_delay_values_from_args(args)
    lamigo_values = lamigo_values_from_args(args)
    out_summary, out_samples = output_paths(args, detectors, t_delay_values, lamigo_values)
    result_entries = []

    for delay_index, t_delay_yr in enumerate(t_delay_values):
        summary, arrays = run_case(
            detectors,
            args.R0,
            args.Tobs,
            t_delay_yr,
            args.rho_thr,
            args.z_max,
            lamigo_values,
            args.n_mc,
            args.n_workers,
            args.seed + delay_index * max(args.n_workers, 1),
        )
        result_entries.append({'summary': summary, 'arrays': arrays})
        print(summary)

    aggregate_summary = {
        'model': 'fullpop_madau',
        'detection_mode': 'shared_space_population_then_multi_LAMIGO_threshold_filtered',
        'space_detector_network': detector_tag(detectors),
        'space_detectors': detectors,
        'followup_detector': 'AMIGO',
        'R0': args.R0,
        'Tobs_space_yr': args.Tobs,
        'Tobs_amigo_yr': T_OBS_AMIGO_YR,
        'LAMIGO_values_m': lamigo_values,
        'Tdelay_values_yr': t_delay_values,
        'rho_thr': args.rho_thr,
        'z_max': args.z_max,
        'n_samples_per_Tdelay': args.n_mc,
        'n_workers': args.n_workers,
        'per_Tdelay': [entry['summary'] for entry in result_entries],
    }

    aggregate_summary['detection_numbers_by_LAMIGO'] = {
        lamigo: [
            {
                'Tdelay_yr': entry['summary']['Tdelay_yr'],
                **entry['summary']['AMIGO_results'][lamigo],
            }
            for entry in result_entries
        ]
        for lamigo in lamigo_values
    }
    if len(t_delay_values) == 1:
        aggregate_summary['Nspace_amigo_by_LAMIGO'] = {
            lamigo: results[0]['Nspace_amigo']
            for lamigo, results in aggregate_summary['detection_numbers_by_LAMIGO'].items()
        }
    else:
        aggregate_summary['Nspace_amigo_by_LAMIGO'] = {
            lamigo: [result['Nspace_amigo'] for result in results]
            for lamigo, results in aggregate_summary['detection_numbers_by_LAMIGO'].items()
        }
    aggregate_summary['n_saved_space_window_positive_total'] = int(
        sum(entry['summary']['n_space_window_positive'] for entry in result_entries)
    )
    save_detectable_samples_multi(out_samples, result_entries, aggregate_summary)
    with open(out_summary, 'w') as f:
        f.write(str(aggregate_summary) + '\n')
        for entry in result_entries:
            f.write(str(entry['summary']) + '\n')
    print('AMIGO detection-number comparison:')
    for lamigo, results in aggregate_summary['detection_numbers_by_LAMIGO'].items():
        for result in results:
            print(
                f"  LAMIGO={lamigo:.0e} m, Tdelay={result['Tdelay_yr']:g} yr: "
                f"Nspace_amigo={result['Nspace_amigo']:.10g}, "
                f"rspace_amigo={result['rspace_amigo']:.10g} yr^-1, "
                f"MC detections={result['n_space_amigo_detected']}"
            )
    print(f'Saved summary to {out_summary}')
    print(f'Saved detectable samples to {out_samples}')
    return aggregate_summary, result_entries

def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
