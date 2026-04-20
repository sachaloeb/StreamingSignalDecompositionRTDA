"""Synthetic signal generators for benchmarking decomposition methods.

All generators return ``np.ndarray`` of length *N* and accept an
optional ``seed`` parameter for reproducibility.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import chirp as scipy_chirp


def two_sinusoids(
    N: int,
    f1: float,
    f2: float,
    A1: float = 1.0,
    A2: float = 1.0,
    fs: float = 1.0,
    snr_db: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Two-sinusoid mixture, optionally corrupted by AWGN.

    Parameters
    ----------
    N : int
        Number of samples.
    f1 : float
        Frequency of the first sinusoid in Hz.
    f2 : float
        Frequency of the second sinusoid in Hz.
    A1 : float, optional
        Amplitude of the first sinusoid.  Default 1.0.
    A2 : float, optional
        Amplitude of the second sinusoid.  Default 1.0.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    snr_db : float or None, optional
        If given, additive white Gaussian noise is added at this SNR.
    seed : int, optional
        Random seed.  Default 42.

    Returns
    -------
    np.ndarray
        Signal of length *N*.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) / fs
    x = A1 * np.sin(2.0 * np.pi * f1 * t) + A2 * np.sin(
        2.0 * np.pi * f2 * t
    )
    if snr_db is not None:
        x = _add_awgn(x, snr_db, rng)
    return x


def chirp_plus_sinusoid(
    N: int,
    f_sin: float,
    f_start: float,
    f_end: float,
    fs: float = 1.0,
    snr_db: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Linear chirp mixed with a stationary sinusoid.

    Parameters
    ----------
    N : int
        Number of samples.
    f_sin : float
        Frequency of the stationary sinusoid in Hz.
    f_start : float
        Start frequency of the chirp in Hz.
    f_end : float
        End frequency of the chirp in Hz.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    snr_db : float or None, optional
        If given, AWGN is added at this SNR.
    seed : int, optional
        Random seed.  Default 42.

    Returns
    -------
    np.ndarray
        Signal of length *N*.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) / fs
    t1 = t[-1]
    x = np.sin(2.0 * np.pi * f_sin * t) + scipy_chirp(
        t, f0=f_start, f1=f_end, t1=t1, method="linear",
    )
    if snr_db is not None:
        x = _add_awgn(x, snr_db, rng)
    return x


def rossler(
    N: int,
    dt: float = 0.01,
    alpha: float = 0.2,
    beta: float = 0.2,
    gamma: float = 3.5,
    seed: int = 42,
) -> np.ndarray:
    """X-component of the Rössler attractor.

    Parameters
    ----------
    N : int
        Number of output samples (after integration and resampling).
    dt : float, optional
        Integration time step.  Default 0.01.
    alpha : float, optional
        Rössler parameter *a*.  Default 0.2.
    beta : float, optional
        Rössler parameter *b*.  Default 0.2.
    gamma : float, optional
        Rössler parameter *c*.  Default 3.5.
    seed : int, optional
        Random seed (used for initial condition perturbation).
        Default 42.

    Returns
    -------
    np.ndarray
        X-component of length *N*.
    """
    rng = np.random.default_rng(seed)

    def _ode(
        _t: float,
        state: np.ndarray,
    ) -> list[float]:
        xr, yr, zr = state
        return [
            -yr - zr,
            xr + alpha * yr,
            beta + zr * (xr - gamma),
        ]

    y0 = [1.0, 1.0, 0.0] + rng.normal(0, 0.01, 3)
    t_span = (0.0, N * dt)
    t_eval = np.linspace(0.0, N * dt, N)

    sol = solve_ivp(
        _ode, t_span, y0, t_eval=t_eval, max_step=dt,
    )
    return sol.y[0]


def component_onset(
    N: int,
    f_steady: float,
    f_onset: float,
    onset_sample: int,
    fs: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Steady sinusoid with a second component appearing mid-signal.

    Parameters
    ----------
    N : int
        Number of samples.
    f_steady : float
        Frequency of the always-present sinusoid in Hz.
    f_onset : float
        Frequency of the component that appears at *onset_sample*.
    onset_sample : int
        Sample index at which the second component begins.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    seed : int, optional
        Random seed (unused here but kept for API consistency).
        Default 42.

    Returns
    -------
    np.ndarray
        Signal of length *N*.
    """
    t = np.arange(N) / fs
    x = np.sin(2.0 * np.pi * f_steady * t)
    onset = np.zeros(N, dtype=np.float64)
    onset[onset_sample:] = np.sin(
        2.0 * np.pi * f_onset * t[onset_sample:]
    )
    return x + onset


def n_sinusoids(
    N: int,
    frequencies: list[float],
    amplitudes: list[float] | None = None,
    fs: float = 1.0,
    snr_db: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Mixture of N sinusoids, optionally corrupted by AWGN.

    Parameters
    ----------
    N : int
        Number of samples.
    frequencies : list[float]
        Frequencies of each sinusoid in Hz.
    amplitudes : list[float] or None, optional
        Amplitude for each sinusoid.  Defaults to 1.0 for all.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    snr_db : float or None, optional
        If given, AWGN is added at this SNR.
    seed : int, optional
        Random seed.  Default 42.

    Returns
    -------
    np.ndarray
        Signal of length *N*.
    """
    rng = np.random.default_rng(seed)
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)
    if len(amplitudes) != len(frequencies):
        raise ValueError("frequencies and amplitudes must have the same length")
    t = np.arange(N) / fs
    x = sum(
        a * np.sin(2.0 * np.pi * f * t)
        for f, a in zip(frequencies, amplitudes)
    )
    x = np.asarray(x, dtype=np.float64)
    if snr_db is not None:
        x = _add_awgn(x, snr_db, rng)
    return x


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _add_awgn(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add white Gaussian noise at a specified SNR.

    Parameters
    ----------
    x : np.ndarray
        Clean signal.
    snr_db : float
        Desired signal-to-noise ratio in dB.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    power_signal = np.dot(x, x) / len(x)
    power_noise = power_signal / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(max(power_noise, 1e-30))
    return x + rng.normal(0.0, noise_std, size=len(x))
