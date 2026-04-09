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

from src.engines.base import DecompositionEngine
from src.engines.ssa import diagonal_averaging, svd_decompose
from src.metrics.stability import nmse as compute_nmse


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

        components: list[np.ndarray] = []
        residual = x.copy()

        for _ in range(self.max_iter):
            M = self._choose_window_length(residual)
            X = self._build_trajectory_matrix(residual, M)
            U, S, Vt = svd_decompose(X)

            freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
            psd = np.abs(np.fft.rfft(residual)) ** 2
            f_max = float(freqs[np.argmax(psd)])

            delta_f = self._fit_gaussian_model(psd, freqs)
            sel = self._select_eigentriples(
                U, S, f_max, delta_f, self.fs, N,
            )

            if len(sel) == 0:
                sel = [0]

            X_sub = np.zeros_like(X)
            for k in sel:
                X_sub += S[k] * np.outer(U[:, k], Vt[k, :])
            g = self._reconstruct_component(X_sub, N)

            g = self._polish(g, residual, N)

            a = self._scale_factor(g, residual)
            g *= a

            components.append(g)
            residual = residual - g

            cur_nmse = compute_nmse(residual, x)
            if cur_nmse < self.nmse_threshold:
                break

        components.append(residual)
        return components

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

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
            [2, N//2].  Falls back to N//3 for near-DC content.
        """
        N = len(residual)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
        psd = np.abs(np.fft.rfft(residual)) ** 2
        f_max = float(freqs[np.argmax(psd)])

        if f_max / self.fs < 1e-3:
            return max(2, N // 3)

        M = int(1.2 * self.fs / f_max)
        return int(np.clip(M, 2, N // 2))

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
        idx = np.arange(N)
        X = np.empty((M, N), dtype=np.float64)
        for i in range(M):
            X[i, :] = x[(idx + i) % N]
        return X

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
        y = np.zeros(N, dtype=np.float64)
        counts = np.zeros(N, dtype=np.float64)
        for i in range(M):
            for j in range(Nc):
                pos = (j + i) % N
                y[pos] += X_sub[i, j]
                counts[pos] += 1.0
        counts = np.maximum(counts, 1e-12)
        return y / counts

    @staticmethod
    def _fit_gaussian_model(
        psd: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """Estimate peak half-width via a 3-Gaussian spectral model.

        Parameters
        ----------
        psd : np.ndarray
            Power spectral density (one-sided).
        freqs : np.ndarray
            Corresponding frequency axis in Hz.

        Returns
        -------
        float
            Estimated half-bandwidth delta_f = 2.5 * sigma_1, where
            sigma_1 is the standard deviation of the dominant Gaussian.
        """

        def _three_gaussians(
            f: np.ndarray,
            a1: float, mu1: float, s1: float,
            a2: float, mu2: float, s2: float,
            a3: float, mu3: float, s3: float,
        ) -> np.ndarray:
            return (
                a1 * np.exp(-0.5 * ((f - mu1) / max(s1, 1e-12)) ** 2)
                + a2 * np.exp(-0.5 * ((f - mu2) / max(s2, 1e-12)) ** 2)
                + a3 * np.exp(-0.5 * ((f - mu3) / max(s3, 1e-12)) ** 2)
            )

        psd_norm = psd / (np.max(psd) + 1e-30)
        f_max_idx = int(np.argmax(psd_norm))
        f_peak = float(freqs[f_max_idx])
        f_range = float(freqs[-1]) if len(freqs) > 1 else 1.0
        sigma_init = max(f_range * 0.05, 1e-6)

        p0 = [
            1.0, f_peak, sigma_init,
            0.3, f_peak * 0.5, sigma_init * 2,
            0.1, f_peak * 1.5, sigma_init * 3,
        ]
        lower = [0] * 9
        upper = [np.inf, np.inf, np.inf] * 3

        try:
            popt, _ = curve_fit(
                _three_gaussians,
                freqs,
                psd_norm,
                p0=p0,
                bounds=(lower, upper),
                maxfev=5000,
            )
            sigma_1 = max(abs(popt[2]), 1e-6)
        except (RuntimeError, ValueError):
            sigma_1 = sigma_init

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
            Original signal length (used for frequency resolution).

        Returns
        -------
        list[int]
            Indices into columns of U / entries of S.
        """
        selected: list[int] = []
        lo = f_max - delta_f
        hi = f_max + delta_f
        for k in range(U.shape[1]):
            freqs_k = np.fft.rfftfreq(U.shape[0], d=1.0 / fs)
            mag_k = np.abs(np.fft.rfft(U[:, k]))
            f_dom = float(freqs_k[np.argmax(mag_k)])
            if lo <= f_dom <= hi:
                selected.append(k)
        return selected

    def _polish(
        self,
        g: np.ndarray,
        residual: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Second-run polishing: re-decompose g and keep only the
        dominant sub-component if it does not increase residual energy.

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
        M = self._choose_window_length(g)
        X2 = self._build_trajectory_matrix(g, M)
        U2, S2, Vt2 = svd_decompose(X2, rank=1)
        X_sub2 = S2[0] * np.outer(U2[:, 0], Vt2[0, :])
        g2 = self._reconstruct_component(X_sub2, N)

        res_old = np.dot(residual - g, residual - g)
        res_new = np.dot(residual - g2, residual - g2)

        if res_new <= res_old:
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
