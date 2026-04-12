"""Rank-1 Incremental SSD engine.

Uses Brand's (2003) rank-1 SVD update to slide the Hankel trajectory
matrix incrementally across windows, avoiding a full SVD rebuild each
stride.  Only the first SSD iteration (dominant component) benefits from
the rank-1-maintained factors; residual iterations fall back to the
standard wrapped-trajectory + full-SVD path inherited from SSD.

Complexity per window (stride s, rank r, embedding M, window N):
  - Cold start  : O(M² N)         — full SVD, same as SSD
  - Warm slide  : O(s · r · N)    — s rank-1 updates, linear in N
  - Residual iters: O((I-1)·M²·N) — inherited SSD path

References
----------
Brand, M. (2003). Fast online SVD revisions for lightweight recommender
systems. *Proc. SIAM International Conference on Data Mining*, 37–46.

Saeed, M., & Alty, S. R. (2020). USSA: A unified singular spectrum
analysis framework with application to real-time data.
*Proc. IEEE ICASSP 2020*, 4837–4841.
"""

from __future__ import annotations

import numpy as np

from src.engines.ssa import diagonal_averaging, svd_decompose
from src.engines.ssd import SSD
from src.engines.svd_update import RankOneUpdater, _build_hankel


class RankOneIncrementalSSD(SSD):
    """SSD with rank-1 trajectory-matrix sliding (Brand 2003 / USSA).

    Each window the dominant-component trajectory matrix is updated via
    ``stride`` calls to :meth:`RankOneUpdater.slide_window` instead of
    being rebuilt from scratch.  Only the *first* SSD iteration uses the
    rank-1-maintained factors; subsequent iterations (residual) use the
    standard wrapped-trajectory path from the parent :class:`SSD`.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    stride : int
        Number of new samples between consecutive windows (must match the
        :class:`~src.streaming.window_manager.WindowManager` stride).
    rank : int, optional
        Number of singular values/vectors maintained by the updater.
        Default 8.
    refresh_every : int, optional
        Full SVD reset interval (in ``slide_window`` calls) to bound
        accumulation error.  Default 50.
    nmse_threshold : float, optional
        SSD stopping criterion.  Default 0.01.
    max_iter : int, optional
        Maximum SSD iterations.  Default 20.
    """

    def __init__(
        self,
        fs: float,
        stride: int,
        rank: int = 8,
        refresh_every: int = 50,
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
        self.stride = stride
        self.rank = rank
        self.refresh_every = refresh_every

        self._updater: RankOneUpdater | None = None
        self._prev_window: np.ndarray | None = None
        self._prev_M: int = 0

    # ------------------------------------------------------------------
    # Override fit()
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> list[np.ndarray]:
        """Decompose *x* using rank-1-updated trajectory factors.

        Parameters
        ----------
        x : np.ndarray
            Window of length N.

        Returns
        -------
        list[np.ndarray]
            [g1, ..., gm, residual].
        """
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        x_energy = np.dot(x, x)
        if x_energy < 1e-30:
            return [x.copy()]

        # Global mean removal (MATLAB line 52)
        x_mean = float(np.mean(x))
        x_zm = x - x_mean
        x_zm_energy = float(np.dot(x_zm, x_zm))
        if x_zm_energy < 1e-30:
            return [x.copy()]

        # ---- choose M & compute PSD on the zero-mean signal ----
        freqs, psd = self._compute_psd(x_zm)
        f_max = float(freqs[np.argmax(psd)])
        M = self._window_length_from_freq(f_max, N)

        # ---- initialise or slide the RankOneUpdater ----
        if self._updater is None or M != self._prev_M:
            self._cold_start(x_zm, M)
        else:
            self._slide(x_zm)

        self._prev_window = x_zm.copy()
        self._prev_M = M

        # ---- SSD extraction loop ----
        return self._ssd_loop(x_zm, x_mean, x_zm_energy, N, M, f_max, freqs, psd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cold_start(self, x: np.ndarray, M: int) -> None:
        """Full SVD cold start; initialises the RankOneUpdater."""
        X_h = _build_hankel(x, M)
        U, S, Vt = np.linalg.svd(X_h, full_matrices=False)
        r = min(self.rank, len(S))
        self._updater = RankOneUpdater(
            U[:, :r], S[:r], Vt[:r, :],
            refresh_every=self.refresh_every,
        )

    def _slide(self, x: np.ndarray) -> None:
        """Slide the updater by ``stride`` samples."""
        N = len(x)
        # The ``stride`` newest samples entered since the last window
        new_samples = x[N - self.stride:]
        for samp in new_samples:
            self._updater.slide_window(float(samp), x)

    def _ssd_loop(
        self,
        x_zm: np.ndarray,
        x_mean: float,
        x_zm_energy: float,
        N: int,
        M: int,
        f_max: float,
        freqs: np.ndarray,
        psd: np.ndarray,
    ) -> list[np.ndarray]:
        """Core SSD extraction loop with rank-1 injection on iteration 0."""
        components: list[np.ndarray] = []
        residual = x_zm.copy()
        prev_nmse = 1.0

        for iteration in range(self.max_iter):
            # Per-iteration mean removal (MATLAB line 70)
            residual -= np.mean(residual)

            if iteration == 0:
                # ---- fast path: rank-1-maintained factors ----
                if f_max / self.fs < 1e-3:
                    g = self._extract_trend(residual)
                else:
                    g = self._extract_iter0(
                        residual, N, M, f_max, freqs, psd,
                    )
            else:
                # ---- residual iterations: full SSD extraction ----
                freqs_r, psd_r = self._compute_psd(residual)
                f_max_r = float(freqs_r[np.argmax(psd_r)])

                if f_max_r / self.fs < 1e-3:
                    g = self._extract_trend(residual)
                else:
                    g = self._extract_component_polished(
                        residual, N, freqs_r, psd_r, f_max_r,
                    )

            a = self._scale_factor(g, residual)
            g *= a
            components.append(g)
            residual = residual - g

            # NMSE: ||residual||² / ||x_zm||²
            cur_nmse = float(np.dot(residual, residual) / x_zm_energy)
            if cur_nmse < self.nmse_threshold:
                break

            # Stagnation detection
            if abs(prev_nmse - cur_nmse) < 1e-5:
                break
            prev_nmse = cur_nmse

        # Add mean back to residual for reconstruction
        residual = residual + x_mean
        components.append(residual)
        return components

    def _extract_iter0(
        self,
        residual: np.ndarray,
        N: int,
        M: int,
        f_max: float,
        freqs: np.ndarray,
        psd: np.ndarray,
    ) -> np.ndarray:
        """Extract the dominant component using rank-1-maintained factors.

        Uses the Hankel-based (U, S, Vt) from the RankOneUpdater.
        Reconstruction via standard diagonal averaging (Hankel formulation).
        """
        U = self._updater.U   # (M, r)
        S = self._updater.S   # (r,)
        Vt = self._updater.Vt  # (r, K)  K = N - M + 1

        delta_f = self._fit_gaussian_model(psd, freqs)
        sel = self._select_eigentriples(U, S, f_max, delta_f, self.fs, N)
        if not sel:
            sel = [0]

        K = N - M + 1
        X_sub = np.zeros((M, K), dtype=np.float64)
        for k in sel:
            if k < len(S) and k < Vt.shape[0]:
                X_sub += S[k] * np.outer(U[:, k], Vt[k, :K])

        g = diagonal_averaging(X_sub)  # length = M + K - 1 = N

        # Polish using parent's method (operates on wrapped trajectory)
        g = self._polish(g, residual, N)
        return g