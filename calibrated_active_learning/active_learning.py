import numpy as np
from .solve_empirical_likelihood import el_newton_lambda, el_weights

def estimate_p(X_2, mu_X, verbose=False):
    lam, info = el_newton_lambda(X_2, mu_X)
    p = el_weights(X_2, mu_X, lam)
    if verbose:
        print(info)
        print("sum p =", p.sum())
        print("moment check (should be close to mu_x):", (p[:, None] * X_2).sum(axis=0)[:5])
    return p

def estimate_pi(uncertainty, sample_budget, p, clip_min=1e-12, clip_max=1-1e-12):
    if p is None:
        p = np.ones(len(uncertainty)) / len(uncertainty)
    pi = p * uncertainty
    pi = pi / pi.sum() * sample_budget
    pi = np.clip(pi, clip_min, clip_max)
    return pi

def sample_by_pi(pi):
    return np.random.binomial(1, pi)

def get_activate_estimator(y_pred, y_true, pi, xi, p=None, estimate_variance=False):
    if p is None:
        p = np.ones(len(y_pred)) / len(y_pred)
    est = np.sum(p * (y_pred + xi/pi*(y_true - y_pred)))
    if not estimate_variance:
        return est
    resid = y_true - y_pred
    var = np.sum(p ** 2 * ((1 - pi) * xi / (pi ** 2)) * (resid ** 2))
    # var = np.sum((p**2) * ((1.0 - pi) / pi) * (resid**2))
    return est, var
