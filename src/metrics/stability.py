"""Stability metrics for evaluating streaming decomposition quality.

Tracks reconstruction fidelity, spectral drift, energy continuity,
and matching confidence over successive analysis windows.
"""

from __future__ import annotations

import numpy as np


def qrf(x_ref: np.ndarray, x_hat: np.ndarray) -> float:
    """Quality of Reconstruction Factor in decibels.

    Parameters
    ----------
    x_ref : np.ndarray
        Reference (original) signal.
    x_hat : np.ndarray
        Reconstructed signal.

    Returns
    -------
    float
        QRF = 20 * log10(||x_ref|| / ||x_ref - x_hat||) dB.
        Returns ``np.inf`` for a perfect reconstruction.
    """
    err_norm = np.linalg.norm(x_ref - x_hat)
    if err_norm < 1e-15:
        return float(np.inf)
    ref_norm = np.linalg.norm(x_ref)
    if ref_norm < 1e-15:
        return 0.0
    return float(20.0 * np.log10(ref_norm / err_norm))


def nmse(residual: np.ndarray, original: np.ndarray) -> float:
    """Normalised mean squared error.

    Parameters
    ----------
    residual : np.ndarray
        Residual signal (original minus reconstructed sum).
    original : np.ndarray
        Original signal.

    Returns
    -------
    float
        ||residual||^2 / ||original||^2.
    """
    orig_energy = np.dot(original, original)
    if orig_energy < 1e-30:
        return 0.0
    return float(np.dot(residual, residual) / orig_energy)


def frequency_drift(freq_trajectory: list[float]) -> float:
    """Variance of a frequency trajectory across windows.

    Parameters
    ----------
    freq_trajectory : list[float]
        Dominant frequency (Hz) recorded at each window.

    Returns
    -------
    float
        ``np.var(freq_trajectory)`` (population variance).
    """
    if len(freq_trajectory) == 0:
        return 0.0
    return float(np.var(freq_trajectory))


def energy_continuity(energy_trajectory: list[float]) -> float:
    """Sum of squared first-differences of an energy trajectory.

    Parameters
    ----------
    energy_trajectory : list[float]
        Energy (e.g. squared norm) at each window.

    Returns
    -------
    float
        Sum of (E_{i+1} - E_i)^2.
    """
    if len(energy_trajectory) < 2:
        return 0.0
    e = np.asarray(energy_trajectory, dtype=np.float64)
    diffs = np.diff(e)
    return float(np.sum(diffs ** 2))


def singular_value_drift(
    S_prev: np.ndarray,
    S_curr: np.ndarray,
) -> float:
    """Frobenius / L2 norm of the change in singular values.

    Parameters
    ----------
    S_prev : np.ndarray
        Singular value vector from the previous window.
    S_curr : np.ndarray
        Singular value vector from the current window.

    Returns
    -------
    float
        ``np.linalg.norm(S_prev - S_curr)``.

    Notes
    -----
    Vectors are zero-padded to equal length when dimensions differ.
    """
    max_len = max(len(S_prev), len(S_curr))
    a = np.zeros(max_len, dtype=np.float64)
    b = np.zeros(max_len, dtype=np.float64)
    a[: len(S_prev)] = S_prev
    b[: len(S_curr)] = S_curr
    return float(np.linalg.norm(a - b))


def matching_confidence(
    cost_matrix: np.ndarray,
    matching: dict[int, int | None],
) -> float:
    """Mean confidence of matched component pairs.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (n_curr, n_prev) with values in [0, 1].
    matching : dict[int, int | None]
        Mapping {curr_idx: prev_idx}. Entries with prev_idx ``None``
        (new components) are excluded.

    Returns
    -------
    float
        Mean of (1 - cost[curr_i, prev_j]) over matched pairs.
        Returns 0.0 when no valid matches exist.
    """
    confidences: list[float] = []
    for curr_i, prev_j in matching.items():
        if prev_j is None:
            continue
        confidences.append(1.0 - cost_matrix[curr_i, prev_j])
    if len(confidences) == 0:
        return 0.0
    return float(np.mean(confidences))
