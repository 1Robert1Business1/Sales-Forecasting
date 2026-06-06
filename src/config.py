"""Central configuration — seeds and shared constants.

The project requires reproducibility end-to-end (see PROJECT_BRIEF.md). Every
script and notebook calls `set_seeds()` before any stochastic operation so the
generator, the SARIMA fits, the Prophet fits, and the LSTM training are all
deterministic from a clean clone.
"""

from __future__ import annotations

import os
import random

RANDOM_SEED: int = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and TensorFlow (if installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
