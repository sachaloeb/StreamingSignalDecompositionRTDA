"""Algorithmically optimized SSD engine.

Inherits the reference :class:`SSD` and overrides only the bottleneck
methods.  The standard SSD in ``ssd.py`` remains untouched.

Optimizations
-------------
1. **FWHM bandwidth estimation** — replaces ``curve_fit`` with a simple
   half-maximum walk + linear interpolation.  O(N) instead of iterative
   nonlinear least-squares.
2. **Moment-based bandwidth estimation** — second central moment of the
   PSD around the peak.
3. **Analytical Jacobian** — when the 3-Gaussian model *is* used, an
   explicit Jacobian halves the number of ``curve_fit`` iterations.
4. **δf caching** — bandwidth is estimated once and reused for both
   polish passes, halving the number of estimation calls.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from src.engines.ssd import SSD


class OptimizedSSD(SSD):
    """Algorithmically optimized SSD.

    Inherits the reference SSD and overrides only the bottleneck
    methods.  The standard SSD remains untouched.

    Optimizations:
    1. FWHM/moment bandwidth estimation replaces curve_fit
    2. Analytical Jacobian when Gaussian model is used
    3. δf caching across polish passes
    4. Batched eigentriple FFT (already in parent — inherited)

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    spectral_method : str, optional
        Bandwidth estimation strategy: ``"fwhm"``, ``"moment"``, or
        ``"gaussian"`` (with analytical Jacobian).  Default ``"fwhm"``.
    nmse_threshold : float, optional
        NMSE stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum extraction iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        spectral_method: str = "fwhm",
        nmse_threshold: float = 0.01,
        max_iter: int = 20,
        **kwargs: object,
    ) -> None:
        super().__init__(
            fs=fs,
            nmse_threshold=nmse_threshold,
            max_iter=max_iter,
            **kwargs,
        )
        if spectral_method not in {"fwhm", "moment", "gaussian"}:
            raise ValueError(
                f"Unknown spectral_method '{spectral_method}'. "
                "Choose from 'fwhm', 'moment', 'gaussian'."
            )
        self.spectral_method = spectral_method

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_gaussian_model(
        psd: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """Route to the appropriate bandwidth estimation method.

        Note: because the parent declares this as ``@staticmethod``, we
        must match the signature.  The ``spectral_method`` attribute is
        not available here, so this static version falls back to FWHM.
        Instance-level dispatch happens via ``_extract_component_polished``
        which calls the correct method directly.
        """
        return OptimizedSSD._estimate_bandwidth_fwhm(psd, freqs)

    # ------------------------------------------------------------------
    # Override: two-pass polish with δf caching + method dispatch
    # ------------------------------------------------------------------

    def _extract_component_polished(
        self,
        residual: np.ndarray,
        N: int,
        freqs: np.ndarray,
        psd: np.ndarray,
        f_max: float,
    ) -> np.ndarray:
        """Full component extraction with two-pass polish loop.

        Identical logic to the parent but:
        - Calls the selected bandwidth estimation method
        - Computes δf ONCE and reuses it for both passes
        """
        # Bandwidth estimation — ONCE (cached for both passes)
        if self.spectral_method == "fwhm":
            delta_f = self._estimate_bandwidth_fwhm(psd, freqs)
        elif self.spectral_method == "moment":
            delta_f = self._estimate_bandwidth_moment(psd, freqs)
        elif self.spectral_method == "gaussian":
            delta_f = self._fit_gaussian_with_jacobian(psd, freqs)
        else:
            delta_f = SSD._fit_gaussian_model(psd, freqs)

        M = self._window_length_from_freq(f_max, N)

        v2 = residual.copy()
        r: np.ndarray | None = None

        for cont in range(2):
            v2 = v2 - np.mean(v2)

            X = self._build_trajectory_matrix(v2, M)
            U, S, Vt = self._decompose_trajectory(X)

            sel = self._select_eigentriples(
                U, S, f_max, delta_f, self.fs, N,
            )
            if len(sel) == 0:
                sel = [0]

            X_sub = np.zeros_like(X)
            for k in sel:
                X_sub += S[k] * np.outer(U[:, k], Vt[k, :])

            # Save pass-1 result before overwriting
            if cont == 1:
                vr = r

            r = self._reconstruct_component(X_sub, N)

            # Convergence condition on second pass
            if cont == 1 and np.dot(r, residual - r) < 0:
                r = vr

            v2 = r

        return r

    # ------------------------------------------------------------------
    # Method 1: FWHM bandwidth estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_bandwidth_fwhm(
        psd: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """Estimate peak half-width via Full Width at Half Maximum.

        Walks left and right from the PSD peak until the amplitude
        drops below half-maximum, using linear interpolation at the
        crossings for sub-bin accuracy.

        Parameters
        ----------
        psd : np.ndarray
            Power spectral density.
        freqs : np.ndarray
            Frequency axis in Hz.

        Returns
        -------
        float
            Estimated half-bandwidth δf = 2.5 * σ₁.
        """
        psd = np.asarray(psd, dtype=np.float64).ravel()
        freqs = np.asarray(freqs, dtype=np.float64).ravel()

        if len(psd) < 2:
            return 2.5 * (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        peak_idx = int(np.argmax(psd))
        peak_val = psd[peak_idx]

        if peak_val < 1e-30:
            # Flat / zero PSD
            return 2.5 * freq_resolution

        half_max = peak_val * 0.5

        # Walk left
        left = peak_idx
        while left > 0 and psd[left] >= half_max:
            left -= 1

        if left > 0 and psd[left] < half_max:
            # Linear interpolation at left crossing
            frac = (half_max - psd[left]) / (psd[left + 1] - psd[left] + 1e-30)
            f_left = freqs[left] + frac * (freqs[left + 1] - freqs[left])
        else:
            f_left = freqs[0]

        # Walk right
        right = peak_idx
        while right < len(psd) - 1 and psd[right] >= half_max:
            right += 1

        if right < len(psd) - 1 and psd[right] < half_max:
            # Linear interpolation at right crossing
            frac = (half_max - psd[right]) / (psd[right - 1] - psd[right] + 1e-30)
            f_right = freqs[right] - frac * (freqs[right] - freqs[right - 1])
        else:
            f_right = freqs[-1]

        fwhm = max(f_right - f_left, 0.0)

        # σ₁ from FWHM: FWHM = 2√(2ln2)·σ ≈ 2.355·σ
        sigma_1 = fwhm / 2.355 if fwhm > 0 else freq_resolution

        return 2.5 * max(sigma_1, freq_resolution)

    # ------------------------------------------------------------------
    # Method 2: Moment-based bandwidth estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_bandwidth_moment(
        psd: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """Estimate peak half-width via second central moment.

        Computes the variance of the PSD around its peak within a
        local window, yielding σ₁ as √(variance).

        Parameters
        ----------
        psd : np.ndarray
            Power spectral density.
        freqs : np.ndarray
            Frequency axis in Hz.

        Returns
        -------
        float
            Estimated half-bandwidth δf = 2.5 * σ₁.
        """
        psd = np.asarray(psd, dtype=np.float64).ravel()
        freqs = np.asarray(freqs, dtype=np.float64).ravel()

        if len(psd) < 2:
            return 2.5 * (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        peak_idx = int(np.argmax(psd))

        # Local window: ±10% of Nyquist or ±20 bins, whichever is smaller
        nyquist = freqs[-1]
        half_range_hz = 0.1 * nyquist
        half_range_bins = int(half_range_hz / freq_resolution) if freq_resolution > 0 else 20
        half_range_bins = min(half_range_bins, 20)
        half_range_bins = max(half_range_bins, 1)

        lo = max(0, peak_idx - half_range_bins)
        hi = min(len(psd), peak_idx + half_range_bins + 1)

        weights = psd[lo:hi]
        f_local = freqs[lo:hi]

        total_w = np.sum(weights)
        if total_w < 1e-30:
            return 2.5 * freq_resolution

        mu = np.sum(f_local * weights) / total_w
        var = np.sum((f_local - mu) ** 2 * weights) / total_w
        sigma_1 = float(np.sqrt(max(var, 0.0)))

        return 2.5 * max(sigma_1, freq_resolution)

    # ------------------------------------------------------------------
    # Method 3: Gaussian model with analytical Jacobian
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_gaussian_with_jacobian(
        psd: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """Estimate peak half-width via 3-Gaussian model with analytical Jacobian.

        Same peak-locked 6-parameter model as standard SSD but supplies
        an explicit Jacobian to ``curve_fit``, reducing maxfev and
        speeding convergence.

        Parameters
        ----------
        psd : np.ndarray
            Power spectral density.
        freqs : np.ndarray
            Frequency axis in Hz.

        Returns
        -------
        float
            Estimated half-bandwidth δf = 2.5 * σ₁.
        """
        psd = np.asarray(psd, dtype=np.float64).ravel()
        freqs = np.asarray(freqs, dtype=np.float64).ravel()

        freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        # ---- locate the two highest spectral peaks ----
        peak_idx, _ = find_peaks(psd)
        if len(peak_idx) >= 2:
            peak_heights = psd[peak_idx]
            top2 = np.argsort(peak_heights)[::-1][:2]
            in1 = int(peak_idx[top2[0]])
            in2 = int(peak_idx[top2[1]])
        elif len(peak_idx) == 1:
            in1 = int(peak_idx[0])
            psd_tmp = psd.copy()
            w = max(1, len(psd) // 20)
            psd_tmp[max(0, in1 - w): min(len(psd), in1 + w + 1)] = 0
            in2 = int(np.argmax(psd_tmp))
        else:
            in1 = int(np.argmax(psd))
            psd_tmp = psd.copy()
            w = max(1, len(psd) // 20)
            psd_tmp[max(0, in1 - w): min(len(psd), in1 + w + 1)] = 0
            in2 = int(np.argmax(psd_tmp))

        f_peak1 = float(freqs[in1])
        f_peak2 = float(freqs[in2])

        # ---- estimate σ₁ ----
        tail1 = psd[in1 + 1:]
        drops1 = np.where(tail1 < (2.0 / 3.0) * psd[in1])[0]
        if len(drops1) > 0 and in1 + 1 + drops1[0] < len(freqs):
            estsig1 = abs(freqs[in1] - freqs[in1 + 1 + int(drops1[0])])
        else:
            estsig1 = max(abs(f_peak1 - f_peak2) * 0.5, 1e-6)

        # ---- estimate σ₂ ----
        tail2 = psd[in2 + 1:]
        drops2 = np.where(tail2 < (2.0 / 3.0) * psd[in2])[0]
        if len(drops2) > 0 and in2 + 1 + drops2[0] < len(freqs):
            estsig2 = abs(freqs[in2] - freqs[in2 + 1 + int(drops2[0])])
        else:
            estsig2 = max(4.0 * abs(f_peak1 - f_peak2), 1e-6)

        estsig1 = max(estsig1, 1e-6)
        estsig2 = max(estsig2, 1e-6)

        # ---- lock centres ----
        mu1 = f_peak1
        mu2 = f_peak2
        mu3 = 0.5 * (f_peak1 + f_peak2)

        # Pre-compute squared-distance arrays
        d1 = -((freqs - mu1) ** 2)
        d2 = -((freqs - mu2) ** 2)
        d3 = -((freqs - mu3) ** 2)

        def _model(
            f: np.ndarray,
            a1: float, a2: float, a3: float,
            s1: float, s2: float, s3: float,
        ) -> np.ndarray:
            s1 = max(abs(s1), 1e-12)
            s2 = max(abs(s2), 1e-12)
            s3 = max(abs(s3), 1e-12)
            return (
                a1 * np.exp(d1 / (2.0 * s1 ** 2))
                + a2 * np.exp(d2 / (2.0 * s2 ** 2))
                + a3 * np.exp(d3 / (2.0 * s3 ** 2))
            )

        def _jac(
            f: np.ndarray,
            a1: float, a2: float, a3: float,
            s1: float, s2: float, s3: float,
        ) -> np.ndarray:
            s1 = max(abs(s1), 1e-12)
            s2 = max(abs(s2), 1e-12)
            s3 = max(abs(s3), 1e-12)
            G1 = np.exp(d1 / (2.0 * s1 ** 2))
            G2 = np.exp(d2 / (2.0 * s2 ** 2))
            G3 = np.exp(d3 / (2.0 * s3 ** 2))
            return np.column_stack([
                G1,                             # ∂/∂a1
                G2,                             # ∂/∂a2
                G3,                             # ∂/∂a3
                a1 * G1 * (-d1) / s1 ** 3,      # ∂/∂s1
                a2 * G2 * (-d2) / s2 ** 3,      # ∂/∂s2
                a3 * G3 * (-d3) / s3 ** 3,      # ∂/∂s3
            ])

        mid_idx = int(round(0.5 * (in1 + in2)))
        mid_idx = min(mid_idx, len(psd) - 1)

        p0 = [
            psd[in1] / 2.0,
            psd[in2] / 2.0,
            psd[mid_idx] / 4.0,
            estsig1,
            estsig2,
            max(4.0 * abs(f_peak1 - f_peak2), 1e-6),
        ]

        try:
            popt, _ = curve_fit(
                _model, freqs, psd,
                p0=p0,
                jac=_jac,
                bounds=([0] * 6, [np.inf] * 6),
                maxfev=500,
            )
            sigma_1 = max(abs(popt[3]), 1e-6)
        except (RuntimeError, ValueError):
            sigma_1 = max(estsig1, 1e-6)

        return 2.5 * sigma_1