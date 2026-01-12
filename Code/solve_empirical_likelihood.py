import numpy as np


def el_newton_lambda(
    X: np.ndarray,
    mu_x: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
    ridge: float = 1e-10,
    min_denom: float = 1e-12,
    backtrack: bool = True,
    verbose: bool = False,
):
    """
    Solve EL calibration Lagrange multiplier lambda via damped Newton.

    Problem:
      g(lambda) = sum_i z_i / (1 + lambda^T z_i) = 0
    where z_i = x_i - mu_x.

    Args:
      X: (n, d) sample matrix (e.g. X for S2)
      mu_x: (d,) target mean of X (population mean or HT estimate)
      max_iter, tol: Newton stopping criteria on ||g||_2
      ridge: small ridge added to (-J) for numerical stability
      min_denom: enforce 1 + lambda^T z_i >= min_denom
      backtrack: do line search to keep feasibility and decrease ||g||
    Returns:
      lam: (d,)
      info: dict with convergence info
    """
    X = np.asarray(X, dtype=float)
    mu_x = np.asarray(mu_x, dtype=float)
    n, d = X.shape

    Z = X - mu_x[None, :]  # (n, d)

    lam = np.zeros(d, dtype=float)

    def compute_g_and_J(lam_vec):
        # denom: (n,)
        denom = 1.0 + Z @ lam_vec
        # feasibility check
        if np.min(denom) <= 0:
            return None, None, denom
        inv = 1.0 / denom                      # (n,)
        g = (Z * inv[:, None]).sum(axis=0)     # (d,)

        inv2 = inv * inv                       # 1/(denom^2)
        # -J = sum z z^T / denom^2  is PSD
        # Compute A = -J = Z^T diag(inv2) Z
        # Efficient: (Z * inv2[:,None]).T @ Z
        A = (Z * inv2[:, None]).T @ Z          # (d, d) = -J
        return g, A, denom

    g, A, denom = compute_g_and_J(lam)
    if g is None:
        raise RuntimeError("Initial lambda is infeasible, this should not happen with lambda=0.")

    g_norm0 = np.linalg.norm(g)
    if verbose:
        print(f"[init] ||g||={g_norm0:.3e}, min_denom={np.min(denom):.3e}")

    for it in range(1, max_iter + 1):
        g, A, denom = compute_g_and_J(lam)
        if g is None:
            raise RuntimeError("Infeasible lambda encountered unexpectedly.")

        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            return lam, {"converged": True, "iter": it - 1, "g_norm": g_norm, "min_denom": float(np.min(denom))}

        # Newton step: lam_new = lam - J^{-1} g
        # But J = -A, so step = (-A)^{-1} g = -A^{-1} g
        # Hence lam_new = lam - (-A^{-1} g) = lam + A^{-1} g
        # Solve (A + ridge I) step = g
        step = np.linalg.solve(A + ridge * np.eye(d), g)

        # Damped update with feasibility + (optional) decrease in ||g||
        t = 1.0
        if backtrack:
            # Ensure denom positivity and try to reduce ||g||
            g_norm_curr = g_norm
            for _ in range(50):
                lam_try = lam + t * step
                g_try, A_try, denom_try = compute_g_and_J(lam_try)
                if g_try is None or np.min(denom_try) < min_denom:
                    t *= 0.5
                    continue
                if np.linalg.norm(g_try) <= (1.0 - 1e-4 * t) * g_norm_curr:
                    lam = lam_try
                    break
                t *= 0.5
            else:
                # If line search fails, still take the largest feasible step
                # (or raise an error if even tiny step is infeasible)
                lam = lam + t * step
        else:
            lam = lam + step

        if verbose:
            g_new, _, denom_new = compute_g_and_J(lam)
            print(f"[iter {it:02d}] t={t:.3e} ||g||={np.linalg.norm(g_new):.3e} min_denom={np.min(denom_new):.3e}")

    # If reached here: not converged
    g, _, denom = compute_g_and_J(lam)
    return lam, {"converged": False, "iter": max_iter, "g_norm": float(np.linalg.norm(g)), "min_denom": float(np.min(denom))}


def el_weights(X: np.ndarray, mu_x: np.ndarray, lam: np.ndarray):
    Z = X - mu_x[None, :]
    denom = 1.0 + Z @ lam
    p = (1.0 / X.shape[0]) * (1.0 / denom)
    return p


if __name__ == "__main__":
    
    # Fake data
    rng = np.random.default_rng(0)
    n, d = 5000, 10
    X = rng.normal(size=(n, d))

    # target mean (pretend this is population mean, or HT estimate)
    mu_x = np.zeros(d)
    mu_x[0] = 0.4  # shift

    lam, info = el_newton_lambda(X, mu_x, verbose=True)
    p = el_weights(X, mu_x, lam)

    print(info)
    print("sum p =", p.sum())
    print("moment check (should be close to mu_x):", (p[:, None] * X).sum(axis=0)[:5])
