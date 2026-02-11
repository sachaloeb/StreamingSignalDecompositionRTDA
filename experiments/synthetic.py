"""Synthetic signal generators for testing and demos."""

from typing import Optional

import numpy as np


def generate_synthetic_signal(
    fs: float,
    duration: float,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Generate synthetic signal with known components.
    
    Signal includes:
    - Linear trend
    - Two sinusoids (different frequencies)
    - Chirp (frequency-modulated component)
    - White noise
    
    Args:
        fs: Sampling frequency in Hz.
        duration: Signal duration in seconds.
        noise_level: Standard deviation of white noise (relative to signal amplitude).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (signal, components_dict) where components_dict contains:
        - 'trend': Linear trend component
        - 'sin1': First sinusoid
        - 'sin2': Second sinusoid
        - 'chirp': Chirp component
        - 'noise': Noise component
        - 'fs': Sampling frequency
        - 't': Time vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    n = len(t)
    
    # Linear trend
    trend = 0.5 * t / duration
    
    # Sinusoids
    f1 = 2.0  # Hz
    f2 = 5.0  # Hz
    sin1 = 1.0 * np.sin(2 * np.pi * f1 * t)
    sin2 = 0.8 * np.sin(2 * np.pi * f2 * t + np.pi / 4)
    
    # Chirp: frequency increases linearly from f_chirp_start to f_chirp_end
    f_chirp_start = 1.0
    f_chirp_end = 8.0
    phase = 2 * np.pi * (f_chirp_start * t + (f_chirp_end - f_chirp_start) * t**2 / (2 * duration))
    chirp = 0.6 * np.sin(phase)
    
    # White noise
    noise = noise_level * np.random.randn(n)
    
    # Total signal
    signal = trend + sin1 + sin2 + chirp + noise
    
    components = {
        "trend": trend,
        "sin1": sin1,
        "sin2": sin2,
        "chirp": chirp,
        "noise": noise,
        "fs": fs,
        "t": t,
    }
    
    return signal, components
