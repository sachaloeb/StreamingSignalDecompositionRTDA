"""Utility functions for seeding, logging, and helpers."""

import logging
import random
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration.
    
    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
