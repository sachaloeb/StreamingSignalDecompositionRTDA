"""Singular Spectrum Decomposition (SSD) algorithm.

Implements the iterative, fully-automated SSA-based decomposition
method of Bonizzi et al. (2014).  At each iteration the dominant
spectral component is extracted from the residual via SSA with an
automatically chosen window length, until the normalised mean-squared
error of the residual falls below a threshold.

Reference
---------
Bonizzi, P., Karel, J. M. H., Meste, O., & Peeters, R. L. M. (2014).
Singular spectrum decomposition: A new method for time series
decomposition.  *Advances in Adaptive Data Analysis*, 6(04), 1450011.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, welch

from src.engines.base import DecompositionEngine
from src.engines.ssa import (
    build_trajectory_matrix,
    diagonal_averaging,
    svd_decompose,
)


class SSD(DecompositionEngine):
    """Singular Spectrum Decomposition.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    nmse_threshold : float, optional
        Stopping criterion: NMSE of residual relative to the original
        signal.  Default 0.01.
    max_iter : int, optional
        Maximum number of extraction iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        nmse_threshold: float = 0.01,
        max_iter: int = 20,
        **kwargs: object,
    ) -> None:
        super().__init__(fs=fs, **kwargs)
        self.nmse_threshold = nmse_threshold
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* into SSD components.

        Parameters
        ----------
        x : np.ndarray
            Input signal of length N.

        Returns
        -------
        list[np.ndarray]
            [g1, g2, ..., gm, residual] where each g_j is an
            extracted component and the last element is the final
            residual.
        """
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        x_energy = np.dot(x, x)
        if x_energy < 1e-30:
            return [x.copy()]

        # Global mean removal (MATLAB line 52: v = v-mean(v))
        x_mean = float(np.mean(x))
        residual = x - x_mean
        x_zm_energy = float(np.dot(residual, residual))
        if x_zm_energy < 1e-30:
            return [x.copy()]

        components: list[np.ndarray] = []
        prev_nmse = 1.0

        for iteration in range(self.max_iter):
            # Per-iteration mean removal (MATLAB line 70: v = v-mean(v))
            residual -= np.mean(residual)

            freqs, psd = self._compute_psd(residual)
            f_max_idx = int(np.argmax(psd))
            f_max = float(freqs[f_max_idx])

            if iteration == 0 and f_max / self.fs < 1e-3:
                g = self._extract_trend(residual)
            else:
                g = self._extract_component_polished(residual, N, freqs, psd, f_max)

            a = self._scale_factor(g, residual)
            g *= a

            components.append(g)
            residual = residual - g

            # NMSE: ||residual||² / ||x - mean(x)||² (MATLAB line 181)
            cur_nmse = float(np.dot(residual, residual) / x_zm_energy)
            if cur_nmse < self.nmse_threshold:
                break

            # Stagnation detection (MATLAB lines 186-190)
            if abs(prev_nmse - cur_nmse) < 1e-5:
                break
            prev_nmse = cur_nmse

        # Add mean back to residual so sum(components) + residual = x
        residual = residual + x_mean
        components.append(residual)
        return components

    # ------------------------------------------------------------------
    # SVD hook (overridden by IncrementalSSD for warm-start)
    # ------------------------------------------------------------------

    def _decompose_trajectory(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD decomposition of the trajectory matrix.

        Subclasses may override to inject warm-start or randomised SVD.
        """
        return svd_decompose(X)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _compute_psd(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD estimation matching MATLAB's pwelch.

        Parameters
        ----------
        x : np.ndarray
            Input signal.

        Returns
        -------
        freqs : np.ndarray
            Frequency axis in Hz.
        psd : np.ndarray
            One-sided power spectral density.
        """
        N = len(x)
        nperseg = min(N, 256)
        freqs, psd = welch(x, fs=self.fs, nperseg=nperseg, nfft=4096)
        return freqs, psd

    def _extract_trend(
        self,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Extract a trend component using standard (non-wrapped) Hankel.

        Fix 6: At iteration 1 with near-DC content, use the standard
        Hankel matrix with rank-1 SVD (matching MATLAB lines 79-94).

        Parameters
        ----------
        residual : np.ndarray
            Current residual signal.

        Returns
        -------
        np.ndarray
            Trend component of length N.
        """
        N = len(residual)
        M = max(2, N // 3)
        X = build_trajectory_matrix(residual, M)
        U, S, Vt = svd_decompose(X, rank=1)
        X_sub = S[0] * np.outer(U[:, 0], Vt[0, :])
        return diagonal_averaging(X_sub)

    def _extract_component_polished(
        self,
        residual: np.ndarray,
        N: int,
        freqs: np.ndarray,
        psd: np.ndarray,
        f_max: float,
    ) -> np.ndarray:
        """Full component extraction with two-pass polish loop.

        Fix 4: Matches MATLAB's ``for cont = 1:2`` loop — runs the
        full inner extraction twice and checks the convergence
        condition ``dot(r, v - r) > 0`` on the second pass.

        Parameters
        ----------
        residual : np.ndarray
            Current residual signal (the reference ``v``).
        N : int
            Signal length.
        freqs : np.ndarray
            Welch frequency axis.
        psd : np.ndarray
            Welch PSD (used for Gaussian bandwidth estimation).
        f_max : float
            Dominant frequency in Hz.

        Returns
        -------
        np.ndarray
            Extracted component of length N.
        """
        delta_f = self._fit_gaussian_model(psd, freqs)
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

            # Fix 4: convergence condition on second pass
            if cont == 1 and np.dot(r, residual - r) < 0:
                r = vr

            v2 = r

        return r

    def _window_length_from_freq(
        self,
        f_max: float,
        N: int,
    ) -> int:
        """Compute embedding dimension from dominant frequency.

        Fix 5: Upper bound is N // 3 (matching MATLAB line 104).
        """
        if f_max / self.fs < 1e-3:
            return max(2, N // 3)
        M = int(1.2 * self.fs / f_max)
        return int(np.clip(M, 2, N // 3))

    def _choose_window_length(
        self,
        residual: np.ndarray,
    ) -> int:
        """Choose embedding dimension M from the dominant PSD peak.

        Parameters
        ----------
        residual : np.ndarray
            Current residual signal.

        Returns
        -------
        int
            Window length M = int(1.2 * fs / f_max), clamped to
            [2, N//3].  Falls back to N//3 for near-DC content.
        """
        N = len(residual)
        freqs, psd = self._compute_psd(residual)
        f_max = float(freqs[np.argmax(psd)])
        return self._window_length_from_freq(f_max, N)

    @staticmethod
    def _build_trajectory_matrix(
        x: np.ndarray,
        M: int,
    ) -> np.ndarray:
        """Build the *wrapped* trajectory matrix (Bonizzi 2014).

        Parameters
        ----------
        x : np.ndarray
            Signal of length N.
        M : int
            Number of rows (embedding dimension).

        Returns
        -------
        np.ndarray
            Shape (M, N).  Row *i* is *x* circularly shifted by *i*.
        """
        N = len(x)
        # Vectorised: build (M, N) index array in one shot then fancy-index x.
        row_shifts = np.arange(M, dtype=np.intp)[:, None]   # (M, 1)
        col_idx    = np.arange(N, dtype=np.intp)[None, :]   # (1, N)
        return x[(col_idx + row_shifts) % N]                 # (M, N)

    @staticmethod
    def _reconstruct_component(
        X_sub: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Diagonal averaging adapted for the wrapped trajectory matrix.

        Parameters
        ----------
        X_sub : np.ndarray
            Rank-reduced trajectory matrix (M x N).
        N : int
            Original signal length.

        Returns
        -------
        np.ndarray
            Reconstructed component of length N.

        Notes
        -----
        For the wrapped formulation each column index *j* contributes
        to position *(j + i) mod N* in the output.  We average all
        contributions to each index.
        """
        M, Nc = X_sub.shape
        # Vectorised wrapped averaging via np.bincount.
        # For the wrapped trajectory every output position n receives
        # exactly M contributions (one per row), so counts = M everywhere.
        i_idx = np.arange(M, dtype=np.intp)[:, None]  # (M, 1)
        j_idx = np.arange(Nc, dtype=np.intp)[None, :]  # (1, Nc)
        pos = (i_idx + j_idx) % N                       # (M, Nc)
        y = np.bincount(pos.ravel(), weights=X_sub.ravel(), minlength=N)
        return y / M

    @staticmethod
    def _fit_gaussian_model(
        psd: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """Estimate peak half-width via a 3-Gaussian spectral model.

        Fix 1: Uses 6 free parameters (A₁, A₂, A₃, σ₁, σ₂, σ₃) with
        locked centres μ₁, μ₂, μ₃, matching MATLAB's gaussfitSSD.m.

        Parameters
        ----------
        psd : np.ndarray
            Power spectral density (one-sided, from Welch).
        freqs : np.ndarray
            Corresponding frequency axis in Hz.

        Returns
        -------
        float
            Estimated half-bandwidth delta_f = 2.5 * sigma_1, where
            sigma_1 is the standard deviation of the dominant Gaussian.
        """
        psd = np.asarray(psd, dtype=np.float64).ravel()
        freqs = np.asarray(freqs, dtype=np.float64).ravel()

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

        # ---- estimate σ₁: where PSD drops to 2/3 of peak ----
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

        # ---- lock centres (MATLAB gaussfitSSD.m) ----
        mu1 = f_peak1
        mu2 = f_peak2
        mu3 = 0.5 * (f_peak1 + f_peak2)

        # Pre-compute squared-distance arrays
        d1 = -((freqs - mu1) ** 2)
        d2 = -((freqs - mu2) ** 2)
        d3 = -((freqs - mu3) ** 2)

        def _model(f: np.ndarray, a1: float, a2: float, a3: float,
                   s1: float, s2: float, s3: float) -> np.ndarray:
            s1 = max(abs(s1), 1e-12)
            s2 = max(abs(s2), 1e-12)
            s3 = max(abs(s3), 1e-12)
            return (
                a1 * np.exp(d1 / (2.0 * s1 ** 2))
                + a2 * np.exp(d2 / (2.0 * s2 ** 2))
                + a3 * np.exp(d3 / (2.0 * s3 ** 2))
            )

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
                bounds=([0] * 6, [np.inf] * 6),
                maxfev=5000,
            )
            sigma_1 = max(abs(popt[3]), 1e-6)
        except (RuntimeError, ValueError):
            sigma_1 = max(estsig1, 1e-6)

        return 2.5 * sigma_1

    @staticmethod
    def _select_eigentriples(
        U: np.ndarray,
        S: np.ndarray,
        f_max: float,
        delta_f: float,
        fs: float,
        N: int,
    ) -> list[int]:
        """Select eigentriple indices whose dominant frequency lies
        within [f_max - delta_f, f_max + delta_f].

        Fix 3: Zero-pads the FFT of each left singular vector to
        max(N, 4096) for adequate frequency resolution.

        Also ensures the eigentriple with the highest energy at
        f_max is included (matching MATLAB lines 135–147).

        Parameters
        ----------
        U : np.ndarray
            Left singular vectors (M x r).
        S : np.ndarray
            Singular values (r,).
        f_max : float
            Target peak frequency in Hz.
        delta_f : float
            Half-bandwidth in Hz.
        fs : float
            Sampling frequency in Hz.
        N : int
            Original signal length (used for zero-padding).

        Returns
        -------
        list[int]
            Indices into columns of U / entries of S.
        """
        n_fft = max(N, 4096)
        n_et = min(U.shape[1], len(S))
        freqs_k = np.fft.rfftfreq(n_fft, d=1.0 / fs)

        # Compute FFT magnitudes for all eigentriples at once
        mags = np.abs(np.fft.rfft(U[:, :n_et], n=n_fft, axis=0))  # (n_fft//2+1, n_et)

        # Dominant frequency of each eigentriple
        f_doms = freqs_k[np.argmax(mags, axis=0)]  # (n_et,)

        lo = f_max - delta_f
        hi = f_max + delta_f
        selected: list[int] = [
            int(k) for k in range(n_et)
            if lo <= f_doms[k] <= hi
        ]

        # Ensure the eigentriple with max energy at f_max is included
        # (MATLAB lines 135-147)
        idx_fmax = int(np.argmin(np.abs(freqs_k - f_max)))
        energy_at_fmax = mags[idx_fmax, :n_et]
        max_et_at_fmax = int(np.argmax(energy_at_fmax))
        if max_et_at_fmax not in selected and len(selected) > 0:
            selected.insert(0, max_et_at_fmax)

        return selected

    def _polish(
        self,
        g: np.ndarray,
        residual: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Second-run polishing (kept for backward compatibility).

        Note: The main extraction path now uses
        ``_extract_component_polished`` which integrates the two-pass
        polish loop directly.  This method is retained for callers that
        invoke it separately.

        Parameters
        ----------
        g : np.ndarray
            Raw extracted component.
        residual : np.ndarray
            Current residual (before subtracting *g*).
        N : int
            Signal length.

        Returns
        -------
        np.ndarray
            Polished component (may be unchanged).
        """
        # Re-decompose g and check convergence condition
        M = self._choose_window_length(g)
        X2 = self._build_trajectory_matrix(g, M)
        U2, S2, Vt2 = svd_decompose(X2, rank=1)
        X_sub2 = S2[0] * np.outer(U2[:, 0], Vt2[0, :])
        g2 = self._reconstruct_component(X_sub2, N)

        # Fix 4: use paper's convergence condition
        if np.dot(g2, residual - g2) > 0:
            return g2
        return g

    @staticmethod
    def _scale_factor(
        g: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Optimal scale: a = (g^T v) / (g^T g).

        Parameters
        ----------
        g : np.ndarray
            Component estimate.
        v : np.ndarray
            Current residual / reference signal.

        Returns
        -------
        float
            Scalar scaling factor.
        """
        gg = np.dot(g, g)
        if gg < 1e-30:
            return 1.0
        return float(np.dot(g, v) / gg)