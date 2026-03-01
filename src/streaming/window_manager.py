"""Sliding-window manager with circular buffer and stride logic.

Buffers incoming samples and emits complete analysis windows at
configurable stride intervals for the streaming SSD pipeline.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class WindowManager:
    """Manages a circular buffer that yields windows at stride boundaries.

    Parameters
    ----------
    window_len : int
        Number of samples in each analysis window.
    stride : int
        Number of new samples between successive window emissions.
    fs : float, optional
        Sampling frequency in Hz.  Default 1.0.
    """

    def __init__(
        self,
        window_len: int,
        stride: int,
        fs: float = 1.0,
    ) -> None:
        self.window_len = window_len
        self.stride = stride
        self.fs = fs
        self._buffer: deque[float] = deque(maxlen=window_len)
        self._sample_idx: int = 0

    def push(self, sample: float) -> np.ndarray | None:
        """Push a new sample into the buffer.

        Parameters
        ----------
        sample : float
            Incoming scalar sample.

        Returns
        -------
        np.ndarray or None
            The current window as a 1-D array when the buffer is full
            **and** the current sample index is a stride boundary;
            otherwise ``None``.
        """
        self._buffer.append(sample)
        self._sample_idx += 1

        if len(self._buffer) < self.window_len:
            return None

        samples_since_full = (
            self._sample_idx - self.window_len
        )
        if samples_since_full % self.stride == 0:
            return np.array(self._buffer, dtype=np.float64)

        return None

    def reset(self) -> None:
        """Clear the buffer and reset the sample counter."""
        self._buffer.clear()
        self._sample_idx = 0

    @property
    def overlap(self) -> int:
        """Number of overlapping samples between successive windows.

        Returns
        -------
        int
            window_len - stride.
        """
        return self.window_len - self.stride
