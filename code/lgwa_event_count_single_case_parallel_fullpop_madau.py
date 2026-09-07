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
T_OBS_YR = 10.0
RHO_THR = 8.0
Z_MAX = 0.8
T_MERGER_GRID_YR = np.logspace(-4, np.log10(20), 200)

N_MC_TOTAL = 1000
N_WORKERS = os.cpu_count() or 4
SEED_BASE = 12345
Q_MIN = 0.1
Q_MAX = 1.0
MASS_GRID_SIZE = 10000
REDSHIFT_GRID_SIZE = 10000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parallel single-case LGWA Monte Carlo with FullPop mass and Madau redshift models'
    )
    parser.add_argument('--R0', type=float, default=R0)
    parser.add_argument('--Tobs', type=float, default=T_OBS_YR)
    parser.add_argument('--rho-thr', type=float, default=RHO_THR)
    parser.add_argument('--z-max', type=float, default=Z_MAX)
    parser.add_argument('--n-mc', type=int, default=N_MC_TOTAL)
    parser.add_argument('--n-workers', type=int, default=N_WORKERS)
    parser.add_argument('--seed', type=int, default=SEED_BASE)
    parser.add_argument('--out-samples', type=Path, default=None)
    parser.add_argument('--out-summary', type=Path, default=None)
    return parser.parse_args()


def case_tag(r0, t_obs_yr, rho_thr, z_max):
    r0_tag = str(r0).replace('.', 'p')
    tobs_tag = str(t_obs_yr).replace('.', 'p')
    rho_tag = str(rho_thr).replace('.', 'p')
    z_tag = str(z_max).replace('.', 'p')
    return f'fullpop_madau_R0-{r0_tag}_Tobs-{tobs_tag}_rho-{rho_tag}_zmax-{z_tag}'


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
    m1_grid = np.geomspace(m_min+0.001, m_max, MASS_GRID_SIZE)
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
    z = sample_from_pdf_grid(z_grid, weight, size, rng)
    return z


def redshift_weight_integral(z_max):
    z_grid, weight = madau_redshift_weight_grid(z_max)
    return float(scipy.integrate.trapezoid(weight, z_grid))


def det_time_window_bounds(m1, m2, z, observation_years, rho_thr):
    dl_mpc = float(tnf.luminosity_distance_mpc_from_z(float(z)))
    _, rho_curve = tnf.snr_vs_merger_time_lgwa(
        m1,
        m2,
        dl_mpc,
        T_MERGER_GRID_YR,
        observation_years=observation_years,
    )
    crossings = tnf.threshold_crossings(T_MERGER_GRID_YR, rho_curve, rho_thr)
    if len(crossings) < 2:
        return 0.0, np.nan, np.nan
    lower = float(min(crossings[0], crossings[-1]))
    upper = float(max(crossings[0], crossings[-1]))
    return upper - lower, lower, upper


def worker_chunk(args):
    t_obs_yr, rho_thr, z_max, n_samples, seed = args
    rng = np.random.default_rng(seed)
    m1, m2, q = sample_fullpop_masses(n_samples, rng)
    z = sample_madau_redshift(n_samples, rng, z_max)
    windows = np.zeros(n_samples)
    t_merger_min = np.full(n_samples, np.nan)
    t_merger_max = np.full(n_samples, np.nan)
    t_merger = np.full(n_samples, np.nan)

    for i in range(n_samples):
        window, lower, upper = det_time_window_bounds(m1[i], m2[i], z[i], t_obs_yr, rho_thr)
        windows[i] = window
        t_merger_min[i] = lower
        t_merger_max[i] = upper
        if window > 0.0:
            t_merger[i] = rng.uniform(lower, upper)

    return {
        'count': n_samples,
        'm1': m1,
        'm2': m2,
        'q': q,
        'z': z,
        'windows': windows,
        't_merger_min': t_merger_min,
        't_merger_max': t_merger_max,
        't_merger': t_merger,
    }


def run_case(r0, t_obs_yr, rho_thr, z_max, n_mc, n_workers, seed_base):
    n_workers = max(1, n_workers)
    base = n_mc // n_workers
    remainder = n_mc % n_workers
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    cases = [
        (t_obs_yr, rho_thr, z_max, chunk_sizes[i], seed_base + i)
        for i in range(n_workers)
        if chunk_sizes[i] > 0
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(worker_chunk, cases))

    total_count = sum(r['count'] for r in results)
    arrays = {
        name: np.concatenate([r[name] for r in results])
        for name in ['m1', 'm2', 'q', 'z', 'windows', 't_merger_min', 't_merger_max', 't_merger']
    }

    total_weight = redshift_weight_integral(z_max)
    mean_window = float(np.mean(arrays['windows']))
    nspace = r0 * total_weight * mean_window
    rspace = nspace / t_obs_yr
    detectable = arrays['windows'] > 0.0

    summary = {
        'model': 'fullpop_madau',
        'R0': r0,
        'Tobs_yr': t_obs_yr,
        'rho_thr': rho_thr,
        'z_max': z_max,
        'n_samples': total_count,
        'n_workers': n_workers,
        'n_det_window_positive': int(np.count_nonzero(detectable)),
        'redshift_weight_integral_Gpc3': total_weight,
        'mean_window_yr': mean_window,
        'Nspace': nspace,
        'rspace': rspace,
    }
    return summary, arrays


def save_detectable_samples(path, arrays, summary):
    detectable = (
        (arrays['windows'] > 0.0)
        & np.isfinite(arrays['t_merger'])
        & (arrays['t_merger'] >= arrays['t_merger_min'])
        & (arrays['t_merger'] <= arrays['t_merger_max'])
    )
    m1 = arrays['m1'][detectable]
    m2 = arrays['m2'][detectable]
    q = arrays['q'][detectable]
    z = arrays['z'][detectable]
    windows = arrays['windows'][detectable]
    t_min = arrays['t_merger_min'][detectable]
    t_max = arrays['t_merger_max'][detectable]
    t_merger = arrays['t_merger'][detectable]
    chirp_mass = tnf.chirp_mass_source(m1, m2)
    samples = np.column_stack([m1, m2, chirp_mass, q, z, windows, t_min, t_max, t_merger])
    header_lines = [
        'LGWA-detectable FullPop+Madau single-case samples with window_yr > 0',
        f'summary = {summary}',
        'columns: m1_msun m2_msun chirp_mass_msun q redshift window_yr t_merger_min_yr t_merger_max_yr t_merger_yr',
    ]
    np.savetxt(path, samples, header='\n'.join(header_lines), fmt='%.10e')


def main():
    args = parse_args()
    summary, arrays = run_case(
        args.R0,
        args.Tobs,
        args.rho_thr,
        args.z_max,
        args.n_mc,
        args.n_workers,
        args.seed,
    )
    print(summary)

    tag = case_tag(args.R0, args.Tobs, args.rho_thr, args.z_max)
    out_summary = args.out_summary or Path(__file__).with_name(f'lgwa_event_count_{tag}.txt')
    out_samples = args.out_samples or Path(__file__).with_name(f'lgwa_event_count_{tag}_samples.txt')
    out_summary.write_text(str(summary) + '\n')
    save_detectable_samples(out_samples, arrays, summary)
    print(f'Saved summary to {out_summary}')
    print(f'Saved detectable samples to {out_samples}')


if __name__ == '__main__':
    main()
