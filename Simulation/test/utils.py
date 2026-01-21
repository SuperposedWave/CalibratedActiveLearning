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
    5D single-index linear DGP where EL first-moment calibration
    perfectly recovers the population mean.

    - Y = beta^T X + eps
    - S2 selection depends only on the same index beta^T X
    """

    assert dim == 5, "This sanity-check DGP is designed for dim=5."

    rng = np.random.default_rng(seed)

    # -----------------------------
    # Population covariates
    # -----------------------------
    X = rng.normal(size=(N, dim))

    # Fixed beta for reproducibility / clarity
    beta = np.array([2.0, -1.5, 1.0, 0.5, -0.5])

    # Outcome
    eps = rng.normal(scale=1.0, size=N)
    Y = X @ beta + eps

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean()

    # -----------------------------
    # S1: small unbiased sample
    # -----------------------------
    idx1 = rng.choice(N, size=n_1, replace=False)
    X_1, Y_1 = X[idx1], Y[idx1]

    # -----------------------------
    # S2: biased sample (single-index)
    # -----------------------------
    kappa = 2.5  # controls bias strength (2~3 is usually enough)
    u = X @ beta

    prob = 1.0 / (1.0 + np.exp(-kappa * u))
    prob = prob / prob.sum()

    idx2 = rng.choice(N, size=n_2, replace=False, p=prob)
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

# def generate_finite_population(
#     N: int,
#     dim: int,
#     n_1: int,
#     n_2: int,
#     is_linear: bool = True,
#     seed: int = None,
#     return_dict: bool = False,
#     use_quadratic_features: bool = False,
# ):
#     rng = np.random.default_rng(seed)

#     # ---------- Population ----------
#     X = rng.normal(size=(N, 1))           # dim = 1
#     beta = 2.0
#     eps = rng.normal(scale=1.0, size=N)
#     Y = beta * X[:, 0] + eps

#     mu_X = X.mean(axis=0)
#     mu_Y = Y.mean()

#     # ---------- S1: small random sample ----------
#     idx1 = rng.choice(N, size=n_1, replace=False)
#     X_1, Y_1 = X[idx1], Y[idx1]

#     # ---------- S2: biased sample (single index) ----------
#     kappa = 2.5
#     score = kappa * X[:, 0]
#     prob = 1.0 / (1.0 + np.exp(-score))
#     prob /= prob.sum()

#     idx2 = rng.choice(N, size=n_2, replace=False, p=prob)
#     X_2, Y_2 = X[idx2], Y[idx2]

#     if return_dict:
#         return dict(
#             X=X, Y=Y, mu_X=mu_X, mu_Y=mu_Y,
#             X_1=X_1, Y_1=Y_1, X_2=X_2, Y_2=Y_2
#         )
#     else:
#         return mu_X, mu_Y, X_1, Y_1, Y_2, X_2


# def generate_finite_population(
#     N: int,
#     dim: int,
#     n_1: int,
#     n_2: int,
#     is_linear: bool = True,
#     seed: Optional[int] = None,
#     return_dict: bool = False,
#     use_quadratic_features: bool = False,
# ) -> Tuple:
#     """
#     Generate finite population data with two samples.

#     Linear case (is_linear=True):
#       - Outcome: Y = X @ beta + eps
#       - S2 sampling: Scheme A (strongly biased), pi(X) aligned with beta^T X

#     Nonlinear case (is_linear=False):
#       - Outcome: Y = m(X) + eps
#       - S2 sampling: Scheme B (strongly biased), pi(X) aligned with a nonlinear index u(X)
#         that correlates strongly with m(X) but depends only on X (NOT Y).

#     Args/Returns: unchanged from your original version.
#     """
#     rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

#     # Keep a raw copy for DGP + sampling score construction.
#     X_raw = rng.normal(size=(N, dim))

#     # -----------------------------
#     # Outcome model
#     # -----------------------------
#     if is_linear:
#         beta = rng.uniform(-5, 5, size=dim) if seed is not None else np.random.uniform(-5, 5, size=dim)
#         eps = rng.normal(size=N) if seed is not None else np.random.randn(N)
#         Y = X_raw @ beta + eps

#         # Features used downstream (possibly expanded)
#         X = X_raw
#         if use_quadratic_features:
#             X_base = X
#             prod = X_base[:, :, None] * X_base[:, None, :]
#             prod = prod.reshape(N, dim * dim)
#             X = np.concatenate([X_base, prod], axis=1)

#         mu_X = np.mean(X, axis=0)

#     else:
#         if X_raw.shape[1] < 5:
#             raise ValueError("dim must be >= 5 for the custom nonlinear DGP.")
#         sigma = 1.0
#         eps = rng.normal(scale=sigma, size=N) if seed is not None else np.random.randn(N) * sigma

#         # Nonlinear mean function m(X) based on raw features
#         m = (
#             10.0 * np.sin(np.pi * X_raw[:, 0] * X_raw[:, 1])
#             + 20.0 * (X_raw[:, 2] - 0.5) ** 2
#             + 10.0 * X_raw[:, 3]
#             + 5.0 * X_raw[:, 4]
#         )
#         Y = m + eps

#         # Features used downstream (possibly expanded)
#         X = X_raw
#         if use_quadratic_features:
#             X_base = X
#             prod = X_base[:, :, None] * X_base[:, None, :]
#             prod = prod.reshape(N, dim * dim)
#             X = np.concatenate([X_base, prod], axis=1)

#         mu_X = np.mean(X, axis=0)

#     mu_Y = np.mean(Y)

#     # -----------------------------
#     # Sample S1: simple random sampling (unbiased)
#     # -----------------------------
#     idx1 = rng.choice(N, size=n_1, replace=False) if seed is not None else np.random.choice(N, size=n_1, replace=False)
#     X_1, Y_1 = X[idx1], Y[idx1]

#     # -----------------------------
#     # Sample S2: biased sampling based on X (NOT Y)
#     # -----------------------------
#     # We construct sampling scores using X_raw to keep the bias mechanism interpretable/stable
#     # even if X is expanded later.
#     if is_linear:
#         # ===== Scheme A (Linear): align selection with outcome direction beta^T X =====
#         # Controls how biased S2 is. Larger => more biased.
#         kappa = 4.0

#         # Align alpha with beta direction.
#         beta_norm = np.linalg.norm(beta) + 1e-12
#         alpha = beta / beta_norm
#         u = X_raw @ alpha  # strong alignment with E[Y|X]

#         # Use softmax-style weights for stronger, tunable bias than logistic.
#         logw = kappa * u
#         logw = logw - np.max(logw)  # stabilize
#         w = np.exp(logw)
#         prob = w / np.sum(w)

#     else:
#         # ===== Scheme B (Nonlinear): align selection with a nonlinear index u(X) =====
#         # Controls how biased S2 is. Larger => more biased.
#         kappa = 1.0

#         # Build u(X) that correlates strongly with m(X) but depends only on X_raw.
#         # Using components similar to m(X) makes selection bias strongly aligned with outcome.
#         u = (
#             1.0 * np.sin(np.pi * X_raw[:, 0] * X_raw[:, 1])
#             + 2.0 * (X_raw[:, 2] - 0.5) ** 2
#             + 1.0 * X_raw[:, 3]
#             + 0.5 * X_raw[:, 4]
#         )

#         # Softmax-style weights
#         # logw = kappa * u
#         # logw = logw - np.max(logw)  # stabilize
#         # w = np.exp(logw)
#         # prob = w / np.sum(w)
#         kappa = 1.0  # 从 1~3 扫
#         prob = 1 / (1 + np.exp(-kappa * u))
#         prob = prob / prob.sum()

#     idx2 = rng.choice(N, size=n_2, replace=False, p=prob) if seed is not None else np.random.choice(N, size=n_2, replace=False, p=prob)
#     X_2, Y_2 = X[idx2], Y[idx2]

#     if return_dict:
#         return {
#             "X": X,
#             "Y": Y,
#             "mu_X": mu_X,
#             "mu_Y": mu_Y,
#             "X_1": X_1,
#             "Y_1": Y_1,
#             "X_2": X_2,
#             "Y_2": Y_2,
#         }
#     else:
#         return mu_X, mu_Y, X_1, Y_1, Y_2, X_2

# def generate_finite_population(
#     N: int,
#     dim: int,
#     n_1: int,
#     n_2: int,
#     is_linear: bool = True,
#     seed: Optional[int] = None,
#     return_dict: bool = False,
#     use_quadratic_features: bool = True
# ):
#     """
#     DGP tailored for calibrated active learning:
#     - Strong covariate shift between population and S2
#     - Selection bias aligned with outcome-relevant directions
#     - Heteroskedastic noise to make uncertainty informative
#     """

#     if seed is not None:
#         rng = np.random.default_rng(seed)
#     else:
#         rng = np.random.default_rng()

#     # -----------------------------
#     # Population covariates
#     # -----------------------------
#     X = rng.normal(size=(N, dim))

#     # -----------------------------
#     # Outcome model: nonlinear + interaction
#     # -----------------------------
#     m = (
#         2.0 * np.sin(X[:, 0])
#         + 1.5 * (X[:, 1] ** 2)
#         + 1.0 * X[:, 2]
#         + 0.8 * X[:, 0] * X[:, 1]
#     )

#     # -----------------------------
#     # Heteroskedastic noise (key!)
#     # Noise scale depends on X0, X1
#     # -----------------------------
#     sigma = 0.5 + 2.0 * (np.abs(X[:, 0]) > 1.0) + 1.0 * (X[:, 1] > 0)
#     eps = rng.normal(scale=sigma, size=N)

#     Y = m + eps

#     mu_X = X.mean(axis=0)
#     mu_Y = Y.mean()

#     # -----------------------------
#     # Sample S1: small, unbiased
#     # -----------------------------
#     idx1 = rng.choice(N, size=n_1, replace=False)
#     X_1, Y_1 = X[idx1], Y[idx1]

#     # -----------------------------
#     # Sample S2: strongly biased selection
#     # Selection aligns with outcome-relevant directions
#     # -----------------------------
#     score = (
#         3.0 * X[:, 0]
#         + 2.0 * X[:, 1]
#         + 1.5 * (X[:, 0] * X[:, 1])
#         + 0.5 * (X[:, 2] ** 2)
#     )

#     prob = 1 / (1 + np.exp(-score))
#     prob = prob / prob.sum()

#     idx2 = rng.choice(N, size=n_2, replace=False, p=prob)
#     X_2, Y_2 = X[idx2], Y[idx2]

#     if return_dict:
#         return {
#             "X": X,
#             "Y": Y,
#             "mu_X": mu_X,
#             "mu_Y": mu_Y,
#             "X_1": X_1,
#             "Y_1": Y_1,
#             "X_2": X_2,
#             "Y_2": Y_2,
#         }
#     else:
#         return mu_X, mu_Y, X_1, Y_1, Y_2, X_2


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
    var_estimates: list = None,
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
    if var_estimates is not None:
        table.add_column("Var Est", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("Coverage(90%)", justify="right")
    table.add_column("Coverage(95%)", justify="right")
    table.add_column("Coverage(99%)", justify="right")
    
    for idx, (method, bias, var, mse, cov) in enumerate(
        zip(methods, biases, variances, mses, coverages)
    ):
        row = [
            method,
            f"{bias:.6f}",
            f"{var:.6f}",
        ]
        if var_estimates is not None:
            row.append(f"{var_estimates[idx]:.6f}")
        row.append(f"{mse:.6f}")
        row.extend([f"{cov[0.90]:.3f}", f"{cov[0.95]:.3f}", f"{cov[0.99]:.3f}"])
        table.add_row(*row)
    
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
