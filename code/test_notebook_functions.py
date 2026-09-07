import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

GAMMA = np.euler_gamma

plt.rcParams.update({
    "figure.figsize": (7.2, 5.2),
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": ":",
    "mathtext.fontset": "dejavuserif",
})

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

G = 6.67430e-11
c = 299792458.0
M_sun = 1.98847e30
pc = 3.0856775814913673e16
year = 365.25 * 24 * 3600

L = 2.5e9
f_star = c / (2 * np.pi * L)

H0 = 67.66
Omega_m = 0.30966
Omega_lambda = 1 - Omega_m

injected = {
    "alpha": 2.9,
    "m_min": 4.6,
    "m_max": 86.3,
    "delta_m": 4.8,
    "mu_g_low": 9.7,
    "sigma_g_low": 0.7,
    "mu_g_high": 30.7,
    "sigma_g_high": 6.3,
    "lambda_g": 0.4,
    "lambda_g_low": 0.8,
    "gamma": 3.3,
    "beta_q": 1.0,
    "kappa": 2.9,
    "z_p": 2.5,
}


def lisa_response_numerical(f, response_file=Path("R.txt")):
    f = np.asarray(f, dtype=float)
    if not response_file.exists():
        x = f / f_star
        return (3 / 20) / (1 + 0.6 * x**2)
    data = np.loadtxt(response_file)
    x = f / f_star
    log_response = np.interp(
        np.log10(x),
        np.log10(data[:, 0]),
        np.log10(data[:, 1]),
        left=np.log10(data[0, 1]),
        right=np.log10(data[-1, 1]),
    )
    return 10**log_response


def lisa_high_frequency_wiggles(f):
    f = np.asarray(f, dtype=float)
    return np.ones_like(f, dtype=float)


def lisa_instrument_psd(f, use_numerical_response=True, show_transfer_wiggles=True):
    f = np.asarray(f, dtype=float)
    p_oms = (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)
    p_acc = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    noise_michelson = p_oms / L**2 + 2 * (1 + np.cos(f / f_star) ** 2) * p_acc / ((2 * np.pi * f) ** 4 * L**2)
    if use_numerical_response:
        response = lisa_response_numerical(f)
        psd = noise_michelson / (2 * response)
    else:
        psd = noise_michelson / ((3 / 10) / (1 + 0.6 * (f / f_star) ** 2))
    if show_transfer_wiggles:
        psd = psd * lisa_high_frequency_wiggles(f)
    return psd


def lisa_confusion_psd(f, observation_years=4):
    f = np.asarray(f, dtype=float)
    params = {
        0.5: (0.133, 243.0, 482.0, 917.0, 0.00258),
        1.0: (0.171, 292.0, 1020.0, 1680.0, 0.00215),
        2.0: (0.165, 299.0, 611.0, 1340.0, 0.00173),
        4.0: (0.138, -221.0, 521.0, 1680.0, 0.00113),
    }
    nearest = min(params, key=lambda value: abs(value - observation_years))
    alpha, beta, kappa, gamma, f_knee = params[nearest]
    amplitude = 9e-45
    exponent = -f**alpha + beta * f * np.sin(kappa * f)
    return amplitude * f ** (-7 / 3) * np.exp(np.clip(exponent, -100, 100)) * (1 + np.tanh(gamma * (f_knee - f)))


def lisa_psd(f, include_confusion=True, observation_years=4):
    psd = lisa_instrument_psd(f)
    if include_confusion:
        psd = psd + lisa_confusion_psd(f, observation_years=observation_years)
    return psd


def lisa_characteristic_strain(f, include_confusion=True, observation_years=4):
    f = np.asarray(f, dtype=float)
    return np.sqrt(f * lisa_psd(f, include_confusion=include_confusion, observation_years=observation_years))


def taiji_response_numerical(f, response_file=Path("R.txt")):
    f = np.asarray(f, dtype=float)
    if not response_file.exists():
        x = f / ( c / (2 * np.pi * 3e9) )
        return (3 / 20) / (1 + 0.6 * x**2)
    data = np.loadtxt(response_file)
    x = f / ( c / (2 * np.pi * 3e9) )  # Taiji arm length is 3e9 m
    log_response = np.interp(
        np.log10(x),
        np.log10(data[:, 0]),
        np.log10(data[:, 1]),
        left=np.log10(data[0, 1]),
        right=np.log10(data[-1, 1]),
    )
    return 10**log_response


def taiji_instrument_psd(f, use_numerical_response=True):
    f = np.asarray(f, dtype=float)
    L_tj = 3e9  # Taiji arm length in meters
    f_star_tj = c / (2 * np.pi * L_tj)
    p_oms = (8e-12) ** 2 * (1 + (2e-3 / f) ** 4)
    p_acc = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    noise_michelson = p_oms / L_tj**2 + 2 * (1 + np.cos(f / f_star_tj) ** 2) * p_acc / ((2 * np.pi * f) ** 4 * L_tj**2)

    if use_numerical_response:
        response = taiji_response_numerical(f)
        psd = noise_michelson / (2 * response)
    else:
        psd = noise_michelson / ((3 / 10) / (1 + 0.6 * (f / f_star_tj) ** 2))
    return psd


def taiji_confusion_psd(f):
    f = np.asarray(f, dtype=float)
    params = [-85.5448, -3.23671, -1.64187, -1.14711, 0.0325887, 0.187854]
    Sc = np.zeros_like(f, dtype=float)
    noise = np.zeros_like(f, dtype=float)
    mask = (f < 1e-2) & (f > 1e-4)
    for i in range(len(params)):
        noise[mask] += params[i] * (np.log(f[mask]*1e3)) ** i
    Sc[mask] = np.exp(noise[mask])
    return Sc


def taiji_psd(f, include_confusion=True, observation_years=4, use_numerical_response=True):
    psd = taiji_instrument_psd(f, use_numerical_response=use_numerical_response)
    if include_confusion:
        psd = psd + taiji_confusion_psd(f)
    return psd


def taiji_characteristic_strain(f, include_confusion=True, observation_years=4):
    f = np.asarray(f, dtype=float)
    return np.sqrt(f * taiji_psd(f, include_confusion=include_confusion, observation_years=observation_years))


def tianqin_response_numerical(f, response_file=Path("R.txt")):
    f = np.asarray(f, dtype=float)
    if not response_file.exists():
        x = f / ( c / (2 * np.pi * np.sqrt(3.)*1.e8) )
        return (3 / 20) / (1 + 0.6 * x**2)
    data = np.loadtxt(response_file)
    x = f / ( c / (2 * np.pi * np.sqrt(3.)*1.e8) )  # TianQin arm length is np.sqrt(3.)*1.e8 m
    log_response = np.interp(
        np.log10(x),
        np.log10(data[:, 0]),
        np.log10(data[:, 1]),
        left=np.log10(data[0, 1]),
        right=np.log10(data[-1, 1]),
    )
    return 10**log_response


def tianqin_instrument_psd(f, use_numerical_response=True):
    f = np.asarray(f, dtype=float)
    L_tq = np.sqrt(3.)*1.e8  # TianQin arm length in meters
    p_oms = (1e-12) ** 2 * (1 + (2e-3 / f) ** 4)
    p_acc = (1e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    f_star_tq = c / (2 * np.pi * L_tq)
    noise_michelson = p_oms / L_tq**2 + 2 * (1 + np.cos(f / f_star_tq) ** 2) * p_acc / ((2 * np.pi * f) ** 4 * L_tq**2)

    if use_numerical_response:
        response = tianqin_response_numerical(f)
        psd = noise_michelson / (2 * response)
    else:
        psd = noise_michelson / ((3 / 10) / (1 + 0.6 * (f / f_star_tq) ** 2))
    return psd


def tianqin_psd(f, observation_years=4, use_numerical_response=True):
    psd = tianqin_instrument_psd(f, use_numerical_response=use_numerical_response)
    return psd


def tianqin_characteristic_strain(f, observation_years=4):
    f = np.asarray(f, dtype=float)
    return np.sqrt(f * tianqin_psd(f, observation_years=observation_years))


def amigo_psd(f, LAMIGO = 1e7):
    f = np.asarray(f, dtype=float)
    if LAMIGO == 1e7:
        SAMIGOp = 0.14e-28
    elif LAMIGO == 5e7:
        SAMIGOp = (19e-15) ** 2
    f_star_AMIGO = c / (2 * np.pi * LAMIGO)
    Sa = 3e-30 * (1 + (f / 0.3) ** 4)
    psd = (20/3) * (1/LAMIGO)**2 * (1+(f/(1.29*f_star_AMIGO))**2) * (SAMIGOp+4*Sa/(2*np.pi*f)**4)
    return psd


def amigo_characteristic_strain(f, LAMIGO=1e7):
    f = np.asarray(f, dtype=float)
    return np.sqrt(f * amigo_psd(f, LAMIGO=LAMIGO))


def lgwa_psd(f):
    data = np.loadtxt('LGWA_Si_psd.txt')
    response = np.interp(f, data[:, 0], data[:, 1])
    return response


def DECIGO_psd(f):
    fp = 7.36
    noise = 1e-48 * (7.05*(1+(f/fp)**2) + 4.8e-3*f**(-4)/(1+(f/fp)**2) + 5.33e-4*f**(-4))
    
    data = np.loadtxt("R.txt")
    x = f / ( c / (2 * np.pi * 1e6) )  # DECIGO arm length is 1000 km
    log_response = np.interp(
        np.log10(x),
        np.log10(data[:, 0]),
        np.log10(data[:, 1]),
        left=np.log10(data[0, 1]),
        right=np.log10(data[-1, 1]),
    )
    response = 10**log_response
    psd = noise / (2 * response)
    return psd

def DECIGO_characteristic_strain(f):
    f = np.asarray(f, dtype=float)
    return np.sqrt(f * DECIGO_psd(f))

def B_DECIGO_noise_curve(f):
    return 1e-46 * (4.04 + 6.399e-2*f**(-4) + 6.399e-3*f**2)


def e_z(z):
    z = np.asarray(z, dtype=float)
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_lambda)


def comoving_distance_mpc(z, n=4000):
    grid = np.linspace(0.0, z, n)
    integrand = 1.0 / e_z(grid)
    return (c / 1000.0) / H0 * trapezoid(integrand, grid)


def luminosity_distance_mpc_from_z(z):
    return (1 + z) * comoving_distance_mpc(z)


def redshift_from_luminosity_distance(distance_mpc):
    lo, hi = 0.0, 1.0
    while luminosity_distance_mpc_from_z(hi) < distance_mpc:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if luminosity_distance_mpc_from_z(mid) < distance_mpc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chirp_mass_source(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def total_mass_source(m1, m2):
    return m1 + m2


def phenoma_transition_frequencies(m1, m2, z):
    """Ajith et al. (2007) nonspinning PhenomA transition frequencies."""
    total_mass_redshifted = (m1 + m2) * M_sun * (1 + z)
    eta = (m1 * m2) / (m1 + m2) ** 2
    coeffs = {
        "f0": (2.9740e-1, 4.4810e-2, 9.5560e-2),
        "f1": (5.9411e-1, 8.9794e-2, 1.9111e-1),
        "f2": (5.0801e-1, 7.7515e-2, 2.2369e-2),
        "f3": (8.4845e-1, 1.2848e-1, 2.7299e-1),
    }
    mass_time = G * total_mass_redshifted / c**3
    return {
        name: (a * eta**2 + b * eta + c0) / (np.pi * mass_time)
        for name, (a, b, c0) in coeffs.items()
    }


def phenoma_lorentzian(f, f1, f2):
    """Lorentzian ringdown factor in the Ajith et al. PhenomA amplitude."""
    return (1 / (2 * np.pi)) * f2 / ((f - f1) ** 2 + f2**2 / 4)


def phenoma_fourier_amplitude(f, m1, m2, z, luminosity_distance):
    """Ajith et al. (2007) PhenomA amplitude used by Robson+ and cited by Gerosa+."""
    f = np.asarray(f, dtype=float)
    transitions = phenoma_transition_frequencies(m1, m2, z)
    f0, f1, f2, f3 = (transitions[name] for name in ("f0", "f1", "f2", "f3"))
    chirp_mass_redshifted = chirp_mass_source(m1, m2) * (1 + z)
    amplitude0 = (
        np.sqrt(5 / 24)
        * (G * chirp_mass_redshifted * M_sun / c**3) ** (5 / 6)
        * f0 ** (-7 / 6)
        / (np.pi ** (2 / 3) * (luminosity_distance / c))
    )
    w = (np.pi * f2 / 2) * (f0 / f1) ** (2 / 3)

    amplitude = np.zeros_like(f)
    inspiral = f < f0
    merger = (f >= f0) & (f < f1)
    ringdown = (f >= f1) & (f < f3)
    amplitude[inspiral] = amplitude0 * (f[inspiral] / f0) ** (-7 / 6)
    amplitude[merger] = amplitude0 * (f[merger] / f0) ** (-2 / 3)
    amplitude[ringdown] = amplitude0 * w * phenoma_lorentzian(f[ringdown], f1, f2)
    return amplitude


def frequency_before_merger(t_merger, chirp_mass_redshifted):
    """Observed GW frequency t_merger seconds before coalescence, Eq. (6)."""
    t_merger = np.asarray(t_merger, dtype=float)
    t_merger = np.maximum(t_merger, np.finfo(float).tiny)
    mass_time = G * chirp_mass_redshifted * M_sun / c**3
    return (5 / (256 * t_merger)) ** (3 / 8) * mass_time ** (-5 / 8) / np.pi


def snr_vs_merger_time(m1, m2, luminosity_distance_mpc, merger_times_yr, detector='LISA', observation_years=4, LAMIGO=1e7):
    """LISA SNR accumulated during a mission ending before merger, using PhenomA."""
    z = redshift_from_luminosity_distance(luminosity_distance_mpc)
    luminosity_distance = luminosity_distance_mpc * 1e6 * pc
    chirp_mass_redshifted = chirp_mass_source(m1, m2) * (1 + z)
    if detector == "DECIGO":
        f_cut = 100.0
    elif detector == "AMIGO":
        f_cut = 10.0
    else:
        f_cut = 1.0
    observation_time = observation_years * year
    snr = np.zeros_like(merger_times_yr, dtype=float)

    for index, merger_time_yr in enumerate(merger_times_yr):
        merger_time = merger_time_yr * year
        f_start = frequency_before_merger(merger_time, chirp_mass_redshifted)
        if merger_time > observation_time:
            f_stop = frequency_before_merger(merger_time - observation_time, chirp_mass_redshifted)
        else:
            f_stop = f_cut
        f_stop = min(f_stop, f_cut)

        if not (np.isfinite(f_start) and np.isfinite(f_stop)) or f_stop <= f_start:
            continue

        frequencies = np.logspace(np.log10(f_start), np.log10(f_stop), 2000)
        h_tilde = phenoma_fourier_amplitude(frequencies, m1, m2, z, luminosity_distance)
        if detector == "LISA":
            integrand = h_tilde**2 / lisa_psd(frequencies, include_confusion=True, observation_years=observation_years)
        elif detector == "Taiji":
            integrand = h_tilde**2 / taiji_psd(frequencies, include_confusion=True, observation_years=observation_years)
        elif detector == "TianQin":
            integrand = h_tilde**2 / tianqin_psd(frequencies, observation_years=observation_years)
        elif detector == "DECIGO":
            integrand = h_tilde**2 / DECIGO_psd(frequencies)
        elif detector == "AMIGO":
            integrand = h_tilde**2 / amigo_psd(frequencies, LAMIGO=LAMIGO)
        snr[index] = np.sqrt((16 / 5) * trapezoid(integrand, frequencies))

    return z, snr


def snr_vs_merger_time_lgwa(m1, m2, luminosity_distance_mpc, merger_times_yr, observation_years=10):
    """LGWA SNR accumulated during a mission ending before merger, using PhenomA."""
    z = redshift_from_luminosity_distance(luminosity_distance_mpc)
    luminosity_distance = luminosity_distance_mpc * 1e6 * pc
    chirp_mass_redshifted = chirp_mass_source(m1, m2) * (1 + z)
    f_cut = 3.0
    observation_time = observation_years * year
    snr = np.zeros_like(merger_times_yr, dtype=float)

    for index, merger_time_yr in enumerate(merger_times_yr):
        merger_time = merger_time_yr * year
        f_start = frequency_before_merger(merger_time, chirp_mass_redshifted)
        if merger_time > observation_time:
            f_stop = frequency_before_merger(merger_time - observation_time, chirp_mass_redshifted)
        else:
            f_stop = f_cut
        f_stop = min(f_stop, f_cut)

        if not (np.isfinite(f_start) and np.isfinite(f_stop)) or f_stop <= f_start:
            continue

        frequencies = np.logspace(np.log10(f_start), np.log10(f_stop), 2000)
        h_tilde = phenoma_fourier_amplitude(frequencies, m1, m2, z, luminosity_distance)
        integrand = h_tilde**2 / lgwa_psd(frequencies)
        snr[index] = np.sqrt((32 / 25) * trapezoid(integrand, frequencies))

    return z, snr


def threshold_crossings(x, y, threshold):
    x = np.asarray(x)
    y = np.asarray(y)
    crossings = []
    above = y >= threshold
    for i in range(len(x) - 1):
        if above[i] == above[i + 1]:
            continue
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]
        if y2 == y1:
            continue
        frac = (threshold - y1) / (y2 - y1)
        crossings.append(x1 + frac * (x2 - x1))
    return crossings


def smooth_taper(m, m_low, delta_m):
    m = np.asarray(m, dtype=float)
    out = np.zeros_like(m)
    if delta_m <= 0:
        out[m >= m_low] = 1.0
        return out
    mid = (m > m_low) & (m < m_low + delta_m)
    hi = m >= m_low + delta_m
    out[hi] = 1.0
    mprime = m[mid] - m_low
    f = np.exp(delta_m / mprime + delta_m / (mprime - delta_m))
    out[mid] = 1.0 / (1.0 + f)
    return out


def truncated_normal_pdf(x, mu, sigma, low, high=np.inf):
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    if sigma <= 0:
        return pdf
    alpha = (low - mu) / sigma
    beta = (high - mu) / sigma if np.isfinite(high) else np.inf
    cdf_alpha = 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))
    cdf_beta = 1.0 if not np.isfinite(beta) else 0.5 * (1.0 + math.erf(beta / math.sqrt(2.0)))
    norm = sigma * math.sqrt(2.0 * math.pi) * (cdf_beta - cdf_alpha)
    mask = (x >= low) & (x <= high)
    pdf[mask] = np.exp(-0.5 * ((x[mask] - mu) / sigma) ** 2) / norm
    return pdf


def broken_power_law_pdf(m, alpha1, alpha2, m_break, m_low, m_high):
    m = np.asarray(m, dtype=float)
    pdf = np.zeros_like(m)
    mask1 = (m >= m_low) & (m < m_break)
    mask2 = (m >= m_break) & (m < m_high)
    pdf[mask1] = (m[mask1] / m_break) ** (-alpha1)
    pdf[mask2] = (m[mask2] / m_break) ** (-alpha2)
    grid = np.geomspace(m_low, m_high, 4000)
    grid_pdf = np.where(grid < m_break, (grid / m_break) ** (-alpha1), (grid / m_break) ** (-alpha2))
    norm = trapezoid(grid_pdf, grid)
    return pdf / norm


def fullpop40_primary_mass_pdf(m, params):
    alpha1 = params["alpha"]
    alpha2 = params["alpha"]
    m_break = params["mu_g_high"]
    m_low = params["m_min"]
    m_high = params["m_max"]
    lambda_g = params["lambda_g"]
    lambda_g_low = params["lambda_g_low"]
    lambda0 = 1.0 - lambda_g
    lambda1 = lambda_g * lambda_g_low
    lambda2 = lambda_g * (1.0 - lambda_g_low)
    pl = broken_power_law_pdf(m, alpha1, alpha2, m_break, m_low, m_high)
    peak1 = truncated_normal_pdf(m, params["mu_g_low"], params["sigma_g_low"], low=m_low, high=m_high)
    peak2 = truncated_normal_pdf(m, params["mu_g_high"], params["sigma_g_high"], low=m_low, high=m_high)
    taper = smooth_taper(m, m_low, params["delta_m"])
    raw = (lambda0 * pl + lambda1 * peak1 + lambda2 * peak2) * taper
    grid = np.geomspace(m_low, m_high, 5000)
    pl_g = broken_power_law_pdf(grid, alpha1, alpha2, m_break, m_low, m_high)
    peak1_g = truncated_normal_pdf(grid, params["mu_g_low"], params["sigma_g_low"], low=m_low, high=m_high)
    peak2_g = truncated_normal_pdf(grid, params["mu_g_high"], params["sigma_g_high"], low=m_low, high=m_high)
    taper_g = smooth_taper(grid, m_low, params["delta_m"])
    raw_g = (lambda0 * pl_g + lambda1 * peak1_g + lambda2 * peak2_g) * taper_g
    norm = trapezoid(raw_g, grid)
    return raw / norm


def madau_dickinson_rate(z, gamma=3.3, kappa=2.9, zp=2.5, R0=1.0):
    """Madau & Dickinson (2014) star-formation-rate shape, arbitrary normalization.

    The paper compares the BBH redshift evolution against a scaled cosmic SFR density
    with κ_SFR = 2.7. A standard shape is
        psi(z) ∝ (1 + z)^a / (1 + ((1 + z)/c)^b).
    We normalize it to unity at z = 0 for plotting.
    """
    z = np.asarray(z, dtype=float)
    rate = (1.0 + z) ** gamma / (1.0 + ((1.0 + z) / (1.0 + zp)) ** (gamma + kappa))
    return R0 * rate * (1.0 + (1.0 + zp) ** (- gamma - kappa))


def power_law_merger_rate(z, kappa):
    z = np.asarray(z, dtype=float)
    rate = (1.0 + z) ** kappa
    return rate / rate[0]


def Taylor_F2(m1, m2, f, dL, chi1, chi2, t_c=0, phi_c=0):
    '''
    Input parameters:
    m1, m2:                observed component masses (solar masses)
    f:                     frequency (Hz), should be a numpy array or a single value
    dL:              luminosity distance (Mpc)
    chi1, chi2:            aligned components of dimensionless spins of m1, m2
    lambda1, lambda2:      dimensionless tidal deformabilities of m1, m2
    z:                     redshift
    t_c:                   constant time shift, default is 0
    phi_c:                 constant phase shift default is 0

    Output:                the waveform amplitude, the phase, the first derivative of the phase w.r.t. f
    '''
    m1_s = m1 * G * M_sun / c**3          # in sec
    m2_s = m2 * G * M_sun / c**3            # in sec
    eta = m1_s*m2_s/(m1_s+m2_s)**2
    M = eta**(3./5)*(m1_s+m2_s)
    D = dL*1.e6*pc / c          # in sec
    A = M**(5./6)/D * np.pi**(-2./3) * np.sqrt(5/24)
    v = (np.pi * (m1_s+m2_s) * f)**(1./3)

    # compute spin-orbit and spin-spin parameters
    delta = (m1-m2)/(m1+m2)
    chi_s = 0.5*(chi1+chi2)
    chi_a = 0.5*(chi1-chi2)
    beta = (113./12-19./3*eta)*chi_s + 113./12*delta*chi_a
    sigma = eta * ( 721./48*(chi_s**2-chi_a**2) - 247./48*(chi_s**2-chi_a**2) ) + (1-2*eta)*(719./96*(chi_s**2+chi_a**2) - 233./96*(chi_s**2+chi_a**2)) + delta*(719./48*chi_s*chi_a-233./48*chi_s*chi_a)

    # PN coefficients
    a0 = 1
    a1 = 0
    a2 = 20./9 * (743./336 + 11./4 * eta)
    a3 = -4 *(4 * np.pi - beta)
    a4 = 10 * (3058673./1016064 + 5429./1008 * eta + 617./144 * eta**2) - 10*sigma
    a5 = np.pi * (38645. / 756 - 65. / 9 * eta ) * (1 + 3 * np.log(v))
    a6 = (11583231236531. / 4694215680 - 640 * np.pi**2 / 3 - 6848. / 21 * GAMMA) + eta * (-15737765635. / 3048192 + 2255 * np.pi**2 / 12) + eta**2 * 76055. / 1728 - 127825. / 1296 * eta**3 - 6848. / 21 * np.log(4 * v)
    a7 = np.pi * (77096675./254016 + 378515./1512*eta - 74045./756*eta**2)

    sum_v = a0 + a1*v + a2*v**2 + a3*v**3 + a4*v**4 + a5*v**5 + a6*v**6 + a7*v**7

    # the phase
    psi_f = 2*np.pi*f*t_c + phi_c - np.pi/4 + 3./(128*eta*v**5)*sum_v

    hf = A * f**(-7./6) * np.exp(psi_f*1.j)

    return hf
