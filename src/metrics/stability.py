"""Stability metrics for evaluating streaming decomposition quality.

Tracks reconstruction fidelity, spectral drift, energy continuity,
and matching confidence over successive analysis windows.

Metric temporal scopes:
- **intra-window**: qrf — no history required.
- **cross-window**: singular_value_drift, energy_continuity,
  energy_continuity_norm — need t-1.
- **global aggregate**: freq_drift_aggregate — post-hoc over full run.

Notes on energy_continuity_norm
--------------------------------
Formula: abs(E_k - E_{k-1}) / (E_k + E_{k-1} + eps), eps = 1e-12.
Bounded in [0, 1].  Value 0 = no energy change; value 1 = one component
has near-zero energy relative to the other.  This is a normalized,
interpretable replacement for the raw ``energy_continuity`` which
produces values in the 10^5 range on outlier windows (e.g., window 6
= 173541 in results/baseline/metrics.csv) due to lack of normalization.
The legacy ``energy_continuity`` is preserved for backward compatibility
with existing CSVs.
"""

from __future__ import annotations

import numpy as np


def qrf(
    signal: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Quality of Reconstruction Factor (Harmouche et al. 2017, Eq. 7).

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        The original window signal.
    reconstruction : np.ndarray, shape (N,)
        The reconstructed signal (sum of extracted components).

    Returns
    -------
    float
        QRF in dB.  Returns ``np.inf`` when reconstruction is
        perfect.  Requires no history — fully intra-window.

    Notes
    -----
    QRF = 20 * log10(||x|| / ||x - x̂||).
    Denominator guard: if ||x - x̂|| < 1e-12, return ``np.inf``.
    """
    denom = float(np.linalg.norm(signal - reconstruction))
    if denom < 1e-12:
        return float(np.inf)
    return float(
        20.0 * np.log10(np.linalg.norm(signal) / denom)
    )


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

    .. deprecated::
        Use :func:`freq_drift_aggregate` instead, which properly
        handles NaN and enforces a minimum of 2 finite values.
    """
    if len(freq_trajectory) == 0:
        return 0.0
    return float(np.var(freq_trajectory))


def energy_continuity(
    components_curr: list[np.ndarray],
    components_prev: list[np.ndarray] | None,
    matching: dict[int, int | None],
) -> float:
    """Sum of squared energy differences between matched components.

    .. deprecated::
        Raw, unnormalized metric.  Preserved for backward compatibility
        with existing CSVs (e.g. results/baseline/metrics.csv).
        Use :func:`energy_continuity_norm` for a bounded [0, 1] metric.

    Parameters
    ----------
    components_curr : list of np.ndarray
        Extracted components from the current window.
    components_prev : list of np.ndarray or None
        Extracted components from the previous window.
        Pass ``None`` at t=0.
    matching : dict[int, int | None]
        Maps curr_index -> prev_index.  ``None`` values mean
        unmatched (skipped).

    Returns
    -------
    float
        Sigma_k (E_k(t) - E_k(t-1))^2 for matched pairs.
        Returns ``np.nan`` when *components_prev* is ``None``.
        Returns 0.0 when matching is empty or all unmatched.

    Notes
    -----
    E_k(t) = dot(g_k, g_k) (L2 energy of component k at t).
    Cross-window metric.  First computable at window index 1.
    """
    if components_prev is None:
        return float("nan")
    total = 0.0
    for curr_i, prev_j in matching.items():
        if prev_j is None:
            continue
        if (curr_i >= len(components_curr)
                or prev_j >= len(components_prev)):
            continue
        e_curr = float(
            np.dot(
                components_curr[curr_i],
                components_curr[curr_i],
            )
        )
        e_prev = float(
            np.dot(
                components_prev[prev_j],
                components_prev[prev_j],
            )
        )
        total += (e_curr - e_prev) ** 2
    return total


def energy_continuity_norm(
    components_curr: list[np.ndarray],
    components_prev: list[np.ndarray] | None,
    matching: dict[int, int | None],
    eps: float = 1e-12,
) -> float:
    """Mean normalized energy change between matched components.

    Formula: mean_k( |E_k(t) - E_{k-1}| / (E_k(t) + E_{k-1} + eps) )

    Bounded in [0, 1].  Value 0 = no energy change across all matched
    pairs; value approaching 1 = one component has near-zero energy
    relative to its match.  ``eps = 1e-12`` prevents division by zero
    when both energies are effectively zero.

    Parameters
    ----------
    components_curr : list of np.ndarray
        Extracted components from the current window.
    components_prev : list of np.ndarray or None
        Extracted components from the previous window.
        Pass ``None`` at t=0.
    matching : dict[int, int | None]
        Maps curr_index -> prev_index.  ``None`` values mean
        unmatched (skipped).
    eps : float, optional
        Denominator guard.  Default 1e-12.

    Returns
    -------
    float
        Mean normalized absolute energy change over matched pairs.
        Returns ``np.nan`` when *components_prev* is ``None``.
        Returns 0.0 when matching is empty or all unmatched.

    Notes
    -----
    E_k(t) = dot(g_k, g_k) (L2 energy of component k at t).
    Cross-window metric.  First computable at window index 1.
    """
    if components_prev is None:
        return float("nan")
    values: list[float] = []
    for curr_i, prev_j in matching.items():
        if prev_j is None:
            continue
        if (curr_i >= len(components_curr)
                or prev_j >= len(components_prev)):
            continue
        e_curr = float(
            np.dot(components_curr[curr_i], components_curr[curr_i])
        )
        e_prev = float(
            np.dot(components_prev[prev_j], components_prev[prev_j])
        )
        values.append(abs(e_curr - e_prev) / (e_curr + e_prev + eps))
    if not values:
        return 0.0
    return float(np.mean(values))


def singular_value_drift(
    S_curr: np.ndarray,
    S_prev: np.ndarray | None,
) -> float:
    """Frobenius norm between consecutive singular value vectors.

    Parameters
    ----------
    S_curr : np.ndarray
        Diagonal of the current window's singular value matrix
        (1-D array of singular values, or full diagonal matrix).
    S_prev : np.ndarray or None
        Diagonal of the previous window's singular value matrix.
        Pass ``None`` at t=0.

    Returns
    -------
    float
        ||S_t - S_{t-1}||_F.  Returns ``np.nan`` when
        *S_prev* is ``None``.

    Notes
    -----
    Cross-window metric.  First computable at window index 1.
    Arrays are zero-padded to the same length before comparison.
    """
    if S_prev is None:
        return float("nan")
    s_c = np.ravel(S_curr).astype(float)
    s_p = np.ravel(S_prev).astype(float)
    max_len = max(len(s_c), len(s_p))
    s_c = np.pad(s_c, (0, max_len - len(s_c)))
    s_p = np.pad(s_p, (0, max_len - len(s_p)))
    return float(np.linalg.norm(s_c - s_p))


def dominant_frequency(
    component: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
) -> float:
    """Dominant frequency of a single component via Welch PSD.

    Parameters
    ----------
    component : np.ndarray, shape (N,)
        A single extracted component from the current window.
    fs : float
        Sampling frequency in Hz.
    nperseg : int or None
        Welch segment length.  Defaults to
        ``min(len(component), 256)``.

    Returns
    -------
    float
        Frequency (Hz) of the PSD peak.  Returns ``np.nan`` for
        constant or very short signals.

    Notes
    -----
    This is the per-row logged value.  Post-hoc variance across
    windows gives the ``freq_drift`` statistic.
    """
    from scipy.signal import welch as _welch

    n = len(component)
    if n < 4:
        return float("nan")
    seg = min(n, nperseg or 256)
    freqs, psd = _welch(component, fs=fs, nperseg=seg)
    if psd.max() < 1e-20:
        return float("nan")
    return float(freqs[np.argmax(psd)])


def freq_drift_aggregate(
    freq_trajectory: list[float] | np.ndarray,
) -> float:
    """Variance of dominant frequencies across all windows.

    Parameters
    ----------
    freq_trajectory : list of float or np.ndarray
        The column of per-row ``dominant_frequency`` values from
        ``metrics.csv``.

    Returns
    -------
    float
        Var_t[f_max(t)], ignoring NaN values.  Returns ``np.nan``
        if fewer than 2 finite values are present.

    Notes
    -----
    This is a **global aggregate** — call it on the full metrics
    DataFrame after the streaming run completes, not inside the
    per-window loop.
    """
    arr = np.asarray(freq_trajectory, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    return float(np.var(arr, ddof=0))


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
