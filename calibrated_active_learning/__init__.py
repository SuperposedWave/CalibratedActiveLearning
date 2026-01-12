"""
Calibrated Active Learning - Code Package

This package contains implementations for empirical likelihood methods
used in calibrated active learning.
"""

from .solve_empirical_likelihood import el_newton_lambda, el_weights
from .active_learning import estimate_p, estimate_pi, sample_by_pi, get_activate_estimator

__all__ = ['el_newton_lambda', 'el_weights', 'estimate_p', 'estimate_pi', 'sample_by_pi', 'get_activate_estimator']

