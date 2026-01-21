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

def estimate_pi(uncertainty, sample_budget, p, tau=0.0):
    if p is None:
        p = np.ones(len(uncertainty)) / len(uncertainty)
    pi = p * uncertainty
    total = np.sum(pi)
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(len(uncertainty)) * (sample_budget / len(uncertainty))
    pi = pi / total * sample_budget
    if tau > 0.0:
        # pi_unif = np.ones(len(uncertainty)) * (sample_budget / len(uncertainty))
        # pi = (1.0 - tau) * pi + tau * pi_unif
        pi = (1.0 - tau) * pi + tau * p
        pi = pi / pi.sum() * sample_budget
    # Clip after mixing to keep probabilities valid.
    pi = np.clip(pi, 0.0, 1.0)
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
    # sample variance
    # var = np.var(p * (y_pred + xi/pi*(y_true - y_pred)) / p.mean())
    return est, var

