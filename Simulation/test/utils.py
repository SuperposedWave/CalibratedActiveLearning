"""
Utility functions for simulation tests.
"""
import numpy as np
import random
from typing import Dict, Tuple, Optional
from rich.console import Console
from rich.table import Table

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def seed_everything(seed: int, use_torch: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        use_torch: Whether to set torch seeds (if torch is available)
    """
    random.seed(seed)
    np.random.seed(seed)
    if use_torch and HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def generate_finite_population(
    N: int,
    dim: int,
    n_1: int,
    n_2: int,
    is_linear: bool = True,
    seed: Optional[int] = None,
    return_dict: bool = False,
    use_quadratic_features: bool = False,
) -> Tuple:
    """
    Generate finite population data with two samples.
    
    Args:
        N: Population size
        dim: Feature dimension
        n_1: Size of sample S1 (small sample)
        n_2: Size of sample S2 (large sample)
        is_linear: Whether to use linear model (True) or nonlinear model (False)
        seed: Random seed
        return_dict: If True, return dictionary; otherwise return tuple
        use_quadratic_features: If True, use quadratic features for nonlinear case (test.py style)
    
    Returns:
        If return_dict=False: (mu_X, mu_Y, X_1, Y_1, Y_2, X_2)
        If return_dict=True: dict with keys: X, Y, mu_X, mu_Y, X_1, Y_1, X_2, Y_2
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    X = rng.normal(size=(N, dim))
    
    if is_linear:
        if seed is not None:
            beta = rng.uniform(-5, 5, size=dim)
        else:
            beta = np.random.uniform(-5, 5, size=dim)
        noise = rng.normal(size=N) if seed is not None else np.random.randn(N)
        Y = X @ beta + noise + 100
        mu_X = np.mean(X, axis=0)
    else:
        if use_quadratic_features:
            # test.py style: quadratic features
            prod = X[:, :, None] * X[:, None, :]
            prod = prod.reshape(N, dim * dim)
            X = np.concatenate([X, prod], axis=1)
            if seed is not None:
                beta = rng.uniform(-5, 5, size=X.shape[1])
            else:
                beta = np.random.uniform(-5, 5, size=X.shape[1])
            noise = rng.normal(size=N) if seed is not None else np.random.randn(N)
            Y = X @ beta + noise + 100
            mu_X = np.mean(X, axis=0)
        else:
            # test_M_estimator.py style: nonlinear function
            eps = rng.normal(size=N) if seed is not None else np.random.randn(N)
            m = (
                2.0 * np.sin(X[:, 0])
                + 1.5 * (X[:, 1] ** 2)
                + 0.8 * X[:, 2]
                + 0.5 * X[:, 3] * X[:, 4]
            )
            Y = m + eps + 100.0
            mu_X = X.mean(axis=0)
    
    mu_Y = np.mean(Y)
    
    # Sample S1: simple random sampling
    if seed is not None:
        idx1 = rng.choice(N, size=n_1, replace=False)
    else:
        idx1 = np.random.choice(N, size=n_1, replace=False)
    X_1, Y_1 = X[idx1], Y[idx1]
    
    # Sample S2: biased sampling based on X
    score = 1.5 * X[:, 0] + 0.5 * X[:, 1]
    prob = 1 / (1 + np.exp(-score))
    prob = prob / prob.sum()
    if seed is not None:
        idx2 = rng.choice(N, size=n_2, replace=False, p=prob)
    else:
        idx2 = np.random.choice(N, size=n_2, replace=False, p=prob)
    X_2, Y_2 = X[idx2], Y[idx2]
    
    if return_dict:
        return {
            "X": X,
            "Y": Y,
            "mu_X": mu_X,
            "mu_Y": mu_Y,
            "X_1": X_1,
            "Y_1": Y_1,
            "X_2": X_2,
            "Y_2": Y_2,
        }
    else:
        return mu_X, mu_Y, X_1, Y_1, Y_2, X_2


def summarize_metrics(estimates: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute bias, variance, and MSE of estimates.
    
    Args:
        estimates: Array of estimates
        truth: Array of true values
    
    Returns:
        bias, variance, mse
    """
    bias = np.mean(estimates - truth)
    variance = np.var(estimates - truth)
    mse = np.mean((estimates - truth) ** 2)
    return bias, variance, mse


def compute_coverage(
    estimates: np.ndarray,
    truth: np.ndarray,
    variances: np.ndarray,
    z_map: Dict[float, float] = None,
) -> Dict[float, float]:
    """
    Compute coverage rates for different confidence levels.
    
    Args:
        estimates: Array of estimates
        truth: Array of true values
        variances: Array of variance estimates
        z_map: Dictionary mapping confidence levels to z-scores
    
    Returns:
        Dictionary mapping confidence levels to coverage rates
    """
    if z_map is None:
        z_map = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    
    coverage = {}
    for level, z in z_map.items():
        coverage[level] = np.mean(
            (truth >= estimates - z * np.sqrt(variances))
            & (truth <= estimates + z * np.sqrt(variances))
        )
    return coverage


def create_results_table(
    methods: list,
    biases: list,
    variances: list,
    mses: list,
    coverages: list,
    title: str = "Active Estimator Comparison",
) -> Table:
    """
    Create a rich table for displaying results.
    
    Args:
        methods: List of method names
        biases: List of bias values
        variances: List of variance values
        mses: List of MSE values
        coverages: List of coverage dictionaries (each dict has keys 0.90, 0.95, 0.99)
        title: Table title
    
    Returns:
        Rich Table object
    """
    table = Table(title=title)
    table.add_column("Method", style="bold")
    table.add_column("Bias", justify="right")
    table.add_column("Variance", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("Coverage(90%)", justify="right")
    table.add_column("Coverage(95%)", justify="right")
    table.add_column("Coverage(99%)", justify="right")
    
    for method, bias, var, mse, cov in zip(methods, biases, variances, mses, coverages):
        table.add_row(
            method,
            f"{bias:.6f}",
            f"{var:.6f}",
            f"{mse:.6f}",
            f"{cov[0.90]:.3f}",
            f"{cov[0.95]:.3f}",
            f"{cov[0.99]:.3f}",
        )
    
    return table


def fixed_budget_draw(weights: np.ndarray, m: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Draw fixed budget samples based on weights.
    
    Args:
        weights: Sampling weights
        m: Budget size
        rng: Random number generator (optional)
    
    Returns:
        Binary array indicating selected samples
    """
    w = np.maximum(weights, 0.0)
    w = w / w.sum()
    if rng is not None:
        idx = rng.choice(len(w), size=m, replace=False, p=w)
    else:
        idx = np.random.choice(len(w), size=m, replace=False, p=w)
    xi = np.zeros(len(w), dtype=int)
    xi[idx] = 1
    return xi

