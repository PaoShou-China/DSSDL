#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FitzHugh-Nagumo dataset module.
"""

import os
import numpy as np
from ..utils import get_datasets_generic

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from .create_dataset import dt, EPSILON
except ImportError:
    dt = 0.01
    EPSILON = 0.20

from .create_dataset import slow_params_eval


def f(features, slow_params):
    """Vectorized FitzHugh-Nagumo dynamics with fixed epsilon."""
    v = features[:, [0]]
    w = features[:, [1]]
    a, b, i_ext = slow_params
    v_dot = v - (v**3) / 3.0 - w + i_ext
    w_dot = EPSILON * (v + a - b * w)
    return np.concatenate([v_dot, w_dot], axis=1)


def get_datasets(use_normalization=True):
    """Load FHN datasets with optional normalization."""
    return get_datasets_generic(current_dir, use_normalization)


def get_raw_dataset_dict():
    """Load raw (unnormalized) datasets."""
    return get_datasets(use_normalization=False)


dataset_dict = get_datasets(use_normalization=True)
