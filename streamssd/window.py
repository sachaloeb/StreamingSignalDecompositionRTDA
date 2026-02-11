"""Streaming buffer and window extraction utilities."""

from typing import Iterator

import numpy as np


class SlidingWindow:
    """Sliding window extractor for streaming signals.
    
    Extracts overlapping windows from a signal with a fixed window length
    and stride.
    """
    
    def __init__(self, window_length: int, stride: int):
        """Initialize sliding window extractor.
        
        Args:
            window_length: Length of each window (W).
            stride: Stride between windows (s). Must be > 0.
        """
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if window_length <= 0:
            raise ValueError(f"window_length must be > 0, got {window_length}")
        
        self.window_length = window_length
        self.stride = stride
    
    def num_windows(self, signal_length: int) -> int:
        """Compute number of windows that can be extracted from signal.
        
        Args:
            signal_length: Total length of the signal.
            
        Returns:
            Number of windows.
        """
        if signal_length < self.window_length:
            return 0
        return 1 + (signal_length - self.window_length) // self.stride
    
    def extract_windows(self, signal: np.ndarray) -> Iterator[np.ndarray]:
        """Extract sliding windows from signal.
        
        Args:
            signal: 1D signal array.
            
        Yields:
            Window arrays of length window_length.
        """
        signal_length = len(signal)
        num_windows = self.num_windows(signal_length)
        
        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_length
            if end > signal_length:
                break
            yield signal[start:end].copy()
