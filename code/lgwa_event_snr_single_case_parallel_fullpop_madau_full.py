from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import h5py
import importlib.util
import os
import sys
import types
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.special

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent / '.matplotlib-cache'))

module_path = Path(__file__).with_name('test_notebook_functions.py')
spec = importlib.util.spec_from_file_location('tnf', module_path)
tnf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tnf)


def lgwa_psd_from_code_dir(f):
    data = np.loadtxt(Path(__file__).with_name('LGWA_Si_psd.txt'))
    return np.interp(f, data[:, 0], data[:, 1])


tnf.lgwa_psd = lgwa_psd_from_code_dir


def install_gwpopulation_utils_stub():
    gwpopulation_stub = types.ModuleType('gwpopulation')
    utils_stub = types.ModuleType('gwpopulation.utils')
    utils_stub.xp = np

    def powerlaw(xx, alpha, high, low):
        xx = np.asarray(xx)
        if np.asarray(alpha).shape == () and alpha == -1:
            norm = 1.0 / np.log(high / low)
        else:
            norm = (1.0 + alpha) / np.asarray(high ** (1.0 + alpha) - low ** (1.0 + alpha))
        prob = np.power(xx, alpha) * norm
        prob *= (xx <= high) & (xx >= low)
        return prob

    def truncnorm(xx, mu, sigma, high, low):
        xx = np.asarray(xx)
        zz = (xx - mu) / sigma
        aa = (low - mu) / sigma
        bb = (high - mu) / sigma
        log_pdf = -(zz**2) / 2.0 - np.log(2.0 * np.pi) / 2.0 - np.log(sigma)
        norm = scipy.special.ndtr(bb) - scipy.special.ndtr(aa)
        prob = np.exp(log_pdf) / norm
        return np.nan_to_num(prob) * (xx >= low) * (xx <= high)

    utils_stub.powerlaw = powerlaw
    utils_stub.truncnorm = truncnorm
    gwpopulation_stub.utils = utils_stub
    sys.modules.setdefault('gwpopulation', gwpopulation_stub)
    sys.modules.setdefault('gwpopulation.utils', utils_stub)


install_gwpopulation_utils_stub()

pairing_path = Path(__file__).with_name('pairing.py')
pairing_spec = importlib.util.spec_from_file_location('pairing', pairing_path)
pairing = importlib.util.module_from_spec(pairing_spec)
pairing_spec.loader.exec_module(pairing)
sys.modules.setdefault('pairing', pairing)

pdb_path = Path(__file__).with_name('pdb_external.py')
pdb_spec = importlib.util.spec_from_file_location('pdb_external', pdb_path)
pdb_external = importlib.util.module_from_spec(pdb_spec)
pdb_spec.loader.exec_module(pdb_external)

DEFAULT_POPSUMMARY = Path(__file__).resolve().parent.parent / 'data' / 'baseline6_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_medians_popsummary.h5'

T_OBS_YR = 10.0
RHO_THR = 8.0
Z_MAX = 0.5
T_MERGER_MIN_YR = 0.001
T_MERGER_MAX_YR = 20.0
F_MIN = 1.0e-3
F_MAX = 3.0
N_FREQ = 1000

N_MC_TOTAL = 1000
N_WORKERS = os.cpu_count() or 4
SEED_BASE = 12345
Q_MIN = 0.1
Q_MAX = 1.0
MASS_GRID_SIZE = 500
REDSHIFT_GRID_SIZE = 10000

REQUIRED_MASS_KEYS = [
    'A', 'A2', 'NSmin', 'NSmax', 'BHmin', 'BHmax',
    'UPPERmin', 'UPPERmax', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5',
    'alpha_1', 'alpha_2', 'alpha_dip', 'mu1', 'sig1', 'mix1', 'mu2', 'sig2', 'mix2',
    'beta_pair_1', 'beta_pair_2', 'mbreak',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parallel FullPop+Madau LGWA event sampler using joint-polarization SNR threshold'
    )
    parser.add_argument('--Tobs', type=float, default=T_OBS_YR)
    parser.add_argument('--rho-thr', type=float, default=RHO_THR)
    parser.add_argument('--z-max', type=float, default=Z_MAX)
    parser.add_argument('--t-merger-min', type=float, default=T_MERGER_MIN_YR)
    parser.add_argument('--t-merger-max', type=float, default=T_MERGER_MAX_YR)
    parser.add_argument('--n-mc', type=int, default=N_MC_TOTAL)
    parser.add_argument('--n-workers', type=int, default=N_WORKERS)
    parser.add_argument('--n-freq', type=int, default=N_FREQ)
    parser.add_argument('--seed', type=int, default=SEED_BASE)
    parser.add_argument('--popsummary', type=Path, default=DEFAULT_POPSUMMARY)
    parser.add_argument('--out-samples', type=Path, default=None)
    parser.add_argument('--out-summary', type=Path, default=None)
    return parser.parse_args()


def case_tag(t_obs_yr, rho_thr, z_max, seed):
    tobs_tag = str(t_obs_yr).replace('.', 'p')
    rho_tag = str(rho_thr).replace('.', 'p')
    z_tag = str(z_max).replace('.', 'p')
    s_tag = str(seed)
    return f'fullpop_madau_Tobs-{tobs_tag}_rho-{rho_tag}_zmax-{z_tag}_seed-{s_tag}'


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


def load_mean_mass_params(path):
    with h5py.File(path, 'r') as f:
        keys = [item.decode() if isinstance(item, bytes) else str(item) for item in f.attrs['hyperparameters']]
        samples = np.asarray(f['posterior/hyperparameter_samples'], dtype=float)
    missing = [key for key in REQUIRED_MASS_KEYS if key not in keys]
    if missing:
        raise KeyError(f'Missing required mass-model keys in {path}: {missing}')
    return {key: float(np.nanmean(samples[:, keys.index(key)])) for key in REQUIRED_MASS_KEYS}


def sample_fullpop_masses(params, size, rng):
    dist = pdb_external.NotchFilterBinnedPairingMassDistribution(mmin=params['NSmin'], mmax=params['BHmax'])
    mass_grid = np.geomspace(params['NSmin'], params['BHmax'], MASS_GRID_SIZE)
    m1_mesh, m2_mesh = np.meshgrid(mass_grid, mass_grid, indexing='ij')
    dataset = {'mass_1': m1_mesh, 'mass_2': m2_mesh}
    joint = np.asarray(dist(dataset, **{key: params[key] for key in REQUIRED_MASS_KEYS}), dtype=float)
    joint = np.where(np.isfinite(joint), joint, 0.0)
    joint[m2_mesh > m1_mesh] = 0.0
    total = np.sum(joint)
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError('Fullpop mass model has non-positive normalization on the sampling grid')
    probs = (joint / total).ravel()
    chosen = rng.choice(probs.size, size=size, p=probs)
    i, j = np.unravel_index(chosen, joint.shape)
    m1 = mass_grid[i]
    m2 = mass_grid[j]
    return m1, m2, m2 / m1


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


def det_frequency_window(chirp_mass_source, z, t_merger_yr, observation_years):
    chirp_mass_redshifted = chirp_mass_source * (1.0 + z)
    f_start = tnf.frequency_before_merger(t_merger_yr * tnf.year, chirp_mass_redshifted)
    if t_merger_yr > observation_years:
        f_end = tnf.frequency_before_merger((t_merger_yr - observation_years) * tnf.year, chirp_mass_redshifted)
    else:
        f_end = F_MAX
    return float(max(f_start, F_MIN)), float(min(f_end, F_MAX))


def joint_polarization_snr(m1, m2, z, t_merger_yr, cos_iota, observation_years, n_freq):
    dl_mpc = float(tnf.luminosity_distance_mpc_from_z(float(z)))
    chirp_mass = float(tnf.chirp_mass_source(m1, m2))
    f_start, f_end = det_frequency_window(chirp_mass, z, t_merger_yr, observation_years)
    if not (np.isfinite(f_start) and np.isfinite(f_end)) or f_end <= f_start:
        return 0.0, dl_mpc, chirp_mass, f_start, f_end

    frequencies = np.geomspace(f_start, f_end, n_freq)
    h0 = tnf.phenoma_fourier_amplitude(
        frequencies,
        m1,
        m2,
        z,
        dl_mpc * 1.0e6 * tnf.pc,
    )
    a_plus = 0.5 * (1.0 + cos_iota**2)
    a_cross = cos_iota
    polarization_factor = a_plus**2 + a_cross**2
    psd = tnf.lgwa_psd(frequencies)
    integrand = polarization_factor * h0**2 / psd
    snr = np.sqrt(4.0 / 5 * 2 * scipy.integrate.trapezoid(integrand, frequencies))
    return float(snr), dl_mpc, chirp_mass, f_start, f_end


def worker_chunk(args):
    t_obs_yr, rho_thr, z_max, t_min, t_max, n_samples, n_freq, seed, mass_params = args
    rng = np.random.default_rng(seed)
    m1, m2, q = sample_fullpop_masses(mass_params, n_samples, rng)
    z = sample_madau_redshift(n_samples, rng, z_max)
    t_merger = rng.uniform(t_min, t_max, n_samples)
    cos_iota = rng.uniform(-1.0, 1.0, n_samples)

    snr = np.zeros(n_samples)
    dl_mpc = np.zeros(n_samples)
    chirp_mass = np.zeros(n_samples)
    f_start = np.zeros(n_samples)
    f_end = np.zeros(n_samples)

    for i in range(n_samples):
        snr[i], dl_mpc[i], chirp_mass[i], f_start[i], f_end[i] = joint_polarization_snr(
            m1[i],
            m2[i],
            z[i],
            t_merger[i],
            cos_iota[i],
            t_obs_yr,
            n_freq,
        )

    return {
        'count': n_samples,
        'm1': m1,
        'm2': m2,
        'chirp_mass': chirp_mass,
        'q': q,
        'z': z,
        'dl_mpc': dl_mpc,
        't_merger': t_merger,
        'cos_iota': cos_iota,
        'snr': snr,
        'f_start': f_start,
        'f_end': f_end,
        'detected': snr >= rho_thr,
    }


def run_case(t_obs_yr, rho_thr, z_max, t_min, t_max, n_mc, n_workers, n_freq, seed_base, mass_params):
    n_workers = max(1, n_workers)
    base = n_mc // n_workers
    remainder = n_mc % n_workers
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    cases = [
        (t_obs_yr, rho_thr, z_max, t_min, t_max, chunk_sizes[i], n_freq, seed_base + i, mass_params)
        for i in range(n_workers)
        if chunk_sizes[i] > 0
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(worker_chunk, cases))

    total_count = sum(r['count'] for r in results)
    arrays = {
        name: np.concatenate([r[name] for r in results])
        for name in ['m1', 'm2', 'chirp_mass', 'q', 'z', 'dl_mpc', 't_merger', 'cos_iota', 'snr', 'f_start', 'f_end', 'detected']
    }
    detected = arrays['detected']
    summary = {
        'model': 'popsummary_mean_notchfilter_binned_pairing_madau_joint_polarization_snr',
        'Tobs_yr': t_obs_yr,
        'rho_thr': rho_thr,
        'z_max': z_max,
        't_merger_min_yr': t_min,
        't_merger_max_yr': t_max,
        'n_samples': total_count,
        'n_workers': n_workers,
        'n_freq': n_freq,
        'mass_grid_size': MASS_GRID_SIZE,
        'n_snr_above_threshold': int(np.count_nonzero(detected)),
        'detected_fraction': float(np.mean(detected)),
        'max_snr': float(np.max(arrays['snr'])) if total_count else 0.0,
    }
    return summary, arrays


def save_detected_samples(path, arrays, summary):
    detected = arrays['detected']
    samples = np.column_stack([
        arrays['m1'][detected],
        arrays['m2'][detected],
        arrays['chirp_mass'][detected],
        arrays['q'][detected],
        arrays['z'][detected],
        arrays['dl_mpc'][detected],
        arrays['t_merger'][detected],
        arrays['cos_iota'][detected],
        arrays['snr'][detected],
        arrays['f_start'][detected],
        arrays['f_end'][detected],
    ])
    header_lines = [
        'FullPop+Madau LGWA events with joint-polarization SNR above threshold',
        f'summary = {summary}',
        'columns: m1_msun m2_msun chirp_mass_msun q redshift dl_mpc t_merger_yr cos_iota snr f_start_hz f_end_hz',
    ]
    np.savetxt(path, samples, header='\n'.join(header_lines), fmt='%.10e')


def main():
    args = parse_args()
    mass_params = load_mean_mass_params(args.popsummary)
    summary, arrays = run_case(
        args.Tobs,
        args.rho_thr,
        args.z_max,
        args.t_merger_min,
        args.t_merger_max,
        args.n_mc,
        args.n_workers,
        args.n_freq,
        args.seed,
        mass_params,
    )
    summary['popsummary'] = str(args.popsummary)
    summary['mass_params_mean'] = mass_params
    print(summary)

    tag = case_tag(args.Tobs, args.rho_thr, args.z_max, args.seed)
    output_dir = Path(__file__).with_name('lgwa_event_snr')
    out_summary = args.out_summary or output_dir / f'lgwa_event_snr_{tag}_full.txt'
    out_samples = args.out_samples or output_dir / f'lgwa_event_snr_{tag}_samples_full.txt'
    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(out_samples).parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, 'w') as f:
        f.write(str(summary) + '\n')
    save_detected_samples(out_samples, arrays, summary)
    print(f'Saved summary to {out_summary}')
    print(f'Saved SNR-threshold samples to {out_samples}')


if __name__ == '__main__':
    main()
