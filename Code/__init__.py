"""
Calibrated Active Learning - Code Package

This package contains implementations for empirical likelihood methods
used in calibrated active learning.
"""

from .solve_empirical_likelihood import el_newton_lambda, el_weights

__all__ = ['el_newton_lambda', 'el_weights']

