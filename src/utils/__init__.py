"""Utility helpers: plotting, reproducibility, and IO."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, include_tensorflow: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    By default this avoids importing TensorFlow so non-TF notebooks
    (e.g., data pipeline) are not blocked by TF binary issues.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if include_tensorflow:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except Exception:
            # TensorFlow may be unavailable or incompatible in current env.
            pass


__all__ = ["set_seed"]
