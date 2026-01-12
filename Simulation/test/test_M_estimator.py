import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from Code import el_newton_lambda, el_weights

# ============================================================
# Repro
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DGP
# ============================================================

def generate_finite_population(N, dim, n_1, n_2, is_linear=False, seed=0):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(N, dim))
    eps = rng.normal(size=N)

    if is_linear:
        beta = rng.uniform(-2, 2, size=dim)
        m = X @ beta
    else:
        m = (
            2.0 * np.sin(X[:, 0])
            + 1.5 * (X[:, 1] ** 2)
            + 0.8 * X[:, 2]
            + 0.5 * X[:, 3] * X[:, 4]
        )

    Y = m + eps + 100.0
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean()

    idx1 = rng.choice(N, size=n_1, replace=False)
    X_1, Y_1 = X[idx1], Y[idx1]

    # biased S2 depends on X
    score = 1.5 * X[:, 0] + 0.5 * X[:, 1]
    prob = 1 / (1 + np.exp(-score))
    prob = prob / prob.sum()
    idx2 = rng.choice(N, size=n_2, replace=False, p=prob)
    X_2, Y_2 = X[idx2], Y[idx2]

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


# ============================================================
# Models (all FCNN)
# ============================================================

class MLP(nn.Module):
    def __init__(self, dim, hidden=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        y = self.net(x)
        return y.squeeze(-1)


def _train_mlp_regression(
    X_tr,
    y_tr,
    *,
    dim,
    hidden=64,
    epochs=200,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
    seed=0,
):
    """Train an MLP regressor with standard MSE."""
    torch.manual_seed(seed)

    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=device)

    model = MLP(dim=dim, hidden=hidden, out_dim=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = X_tr.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            pred = model(X_tr[b])
            loss = torch.mean((pred - y_tr[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def _mlp_predict(model, X, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    return model(X_t).detach().cpu().numpy()


# ============================================================
# Cross-fitted uncertainty u(x) = E|f(x)-Y| | X=x
# (No XGB; all MLP)
# ============================================================

def fit_f_and_u_crossfit_mlp(
    X1,
    Y1,
    X2,
    *,
    K=5,
    f_epochs=250,
    u_epochs=250,
    hidden=64,
    lr=1e-3,
    batch_size=128,
    device="cpu",
    seed=0,
):
    """
    Cross-fitting pipeline:
      1) For each fold: train f on train-fold, get OOF predictions on val-fold
      2) Train u on all X1 with target |Y1 - f_oof(X1)|
      3) Train final f on all X1 and predict f(X2), u(X2)

    Returns:
      f_pseudo_X2: (n2,) pseudo labels
      uhat_X2: (n2,) uncertainty proxy, nonnegative
    """
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    n1, d = X1.shape

    oof_pred = np.zeros(n1, dtype=float)

    # Step A: OOF predictions
    for fold, (tr, va) in enumerate(kf.split(X1)):
        f_fold = _train_mlp_regression(
            X1[tr],
            Y1[tr],
            dim=d,
            hidden=hidden,
            epochs=f_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed + 10 + fold,
        )
        oof_pred[va] = _mlp_predict(f_fold, X1[va], device=device)

    # Step B: train u on |residual|
    res_abs = np.abs(Y1 - oof_pred)
    u_model = _train_mlp_regression(
        X1,
        res_abs,
        dim=d,
        hidden=hidden,
        epochs=u_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed + 1000,
    )

    # Step C: final f for pseudo labels
    f_model = _train_mlp_regression(
        X1,
        Y1,
        dim=d,
        hidden=hidden,
        epochs=f_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed + 2000,
    )

    f_pseudo_X2 = _mlp_predict(f_model, X2, device=device)
    uhat_X2 = _mlp_predict(u_model, X2, device=device)
    uhat_X2 = np.maximum(uhat_X2, 0.0) + 1e-12

    return f_pseudo_X2, uhat_X2


# ============================================================
# Sampling rule pi and draws
# ============================================================

def build_pi_from_weights(w, budget, clip_min=0.01, clip_max=0.95, eps=1e-12):
    w = np.maximum(w, 0.0) + eps
    pi = w / w.sum() * budget
    pi = np.clip(pi, clip_min, clip_max)

    # one rescale step to match expected budget (optional)
    s = pi.sum()
    if s != budget:
        mask = (pi > clip_min + 1e-12) & (pi < clip_max - 1e-12)
        if mask.any():
            target = budget - pi[~mask].sum()
            factor = target / max(pi[mask].sum(), eps)
            pi[mask] = np.clip(pi[mask] * factor, clip_min, clip_max)
    return pi


def poisson_draw(pi, rng):
    return rng.binomial(1, pi)


def fixed_budget_draw(weights, m, rng):
    w = np.maximum(weights, 0.0)
    w = w / w.sum()
    idx = rng.choice(len(w), size=m, replace=False, p=w)
    xi = np.zeros(len(w), dtype=int)
    xi[idx] = 1
    return xi


# ============================================================
# Train: prediction-powered / DR risk with calibration weights p
# ============================================================

def train_active_m_estimator(
    X_2,
    Y_2,
    p,
    pi,
    xi,
    f_pseudo,
    *,
    dim,
    hidden=64,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    w_clip=20.0,
    device="cpu",
    seed=0,
):
    """Train theta (MLP) by minimizing weighted DR risk."""
    torch.manual_seed(seed)

    X2 = torch.tensor(X_2, dtype=torch.float32, device=device)
    Y2 = torch.tensor(Y_2, dtype=torch.float32, device=device)

    p_t = torch.tensor(p, dtype=torch.float32, device=device)
    pi_t = torch.tensor(pi, dtype=torch.float32, device=device)
    xi_t = torch.tensor(xi, dtype=torch.float32, device=device)
    f_t = torch.tensor(f_pseudo, dtype=torch.float32, device=device)

    model = MLP(dim=dim, hidden=hidden, out_dim=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n2 = X_2.shape[0]
    idx = np.arange(n2)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n2, batch_size):
            b = idx[start : start + batch_size]

            xb = X2[b]
            yb = Y2[b]
            pb = p_t[b]
            pib = pi_t[b]
            xib = xi_t[b]
            fb = f_t[b]

            pred = model(xb)

            loss_pseudo = (pred - fb) ** 2
            loss_true = (pred - yb) ** 2

            w = xib / pib
            if w_clip is not None:
                w = torch.clamp(w, max=w_clip)

            dr = loss_pseudo + w * (loss_true - loss_pseudo)
            obj = torch.sum(pb * dr)

            opt.zero_grad()
            obj.backward()
            opt.step()

    return model


# Baseline: ERM on labeled set

def train_erm_on_labeled(
    X_l,
    Y_l,
    *,
    dim,
    hidden=64,
    epochs=80,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
    seed=0,
):
    torch.manual_seed(seed)

    Xl = torch.tensor(X_l, dtype=torch.float32, device=device)
    Yl = torch.tensor(Y_l, dtype=torch.float32, device=device)

    model = MLP(dim=dim, hidden=hidden, out_dim=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = X_l.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            pred = model(Xl[b])
            loss = torch.mean((pred - Yl[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


@torch.no_grad()

def eval_mse(model, X, Y, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    pred = model(X_t)
    return torch.mean((pred - Y_t) ** 2).item()


# ============================================================
# One simulation replicate
# ============================================================

def simulate_once(
    *,
    N=10000,
    dim=10,
    n_1=100,
    n_2=1000,
    budget=500,
    is_linear=False,
    seed=0,
    device="cpu",
    use_fixed_budget=False,
    K=5,
):
    rng = np.random.default_rng(seed)

    data = generate_finite_population(N, dim, n_1, n_2, is_linear=is_linear, seed=seed)
    X, Y = data["X"], data["Y"]
    X_1, Y_1 = data["X_1"], data["Y_1"]
    X_2, Y_2 = data["X_2"], data["Y_2"]
    mu_X = data["mu_X"]

    # standardize Y by S1 stats
    y_mean = Y_1.mean()
    y_std = Y_1.std() + 1e-12
    Y_std = (Y - y_mean) / y_std
    Y1_std = (Y_1 - y_mean) / y_std
    Y2_std = (Y_2 - y_mean) / y_std

    # Step 1: cross-fitted f and u using MLPs (no XGB)
    f_pseudo, uhat = fit_f_and_u_crossfit_mlp(
        X_1,
        Y1_std,
        X_2,
        K=K,
        f_epochs=250,
        u_epochs=250,
        hidden=64,
        lr=1e-3,
        batch_size=128,
        device=device,
        seed=seed,
    )

    # Step 2: EL calibration weights p on S2
    lam, info = el_newton_lambda(X_2, mu_X)
    p = el_weights(X_2, mu_X, lam)
    p0 = np.ones_like(p) / len(p)

    # (A) Calibrated Active: pi ∝ p * uhat
    w_cal = p * uhat
    if use_fixed_budget:
        xi_cal = fixed_budget_draw(w_cal, budget, rng)
        pi_cal = build_pi_from_weights(w_cal, budget)
    else:
        pi_cal = build_pi_from_weights(w_cal, budget)
        xi_cal = poisson_draw(pi_cal, rng)

    model_cal_active = train_active_m_estimator(
        X_2,
        Y2_std,
        p=p,
        pi=pi_cal,
        xi=xi_cal,
        f_pseudo=f_pseudo,
        dim=dim,
        hidden=64,
        epochs=30,
        device=device,
        seed=seed + 3000,
    )

    # (B) Raw Active: pi ∝ uhat, uniform weights
    w_raw = uhat
    if use_fixed_budget:
        xi_raw = fixed_budget_draw(w_raw, budget, rng)
        pi_raw = build_pi_from_weights(w_raw, budget)
    else:
        pi_raw = build_pi_from_weights(w_raw, budget)
        xi_raw = poisson_draw(pi_raw, rng)

    model_raw_active = train_active_m_estimator(
        X_2,
        Y2_std,
        p=p0,
        pi=pi_raw,
        xi=xi_raw,
        f_pseudo=f_pseudo,
        dim=dim,
        hidden=64,
        epochs=30,
        device=device,
        seed=seed + 4000,
    )

    # (C) Calibrated Random: pi constant = budget/n2
    pi_rand = np.ones(n_2) * (budget / n_2)
    pi_rand = np.clip(pi_rand, 0.01, 0.95)
    if use_fixed_budget:
        xi_rand = fixed_budget_draw(np.ones(n_2), budget, rng)
    else:
        xi_rand = poisson_draw(pi_rand, rng)

    model_cal_random = train_active_m_estimator(
        X_2,
        Y2_std,
        p=p,
        pi=pi_rand,
        xi=xi_rand,
        f_pseudo=f_pseudo,
        dim=dim,
        hidden=64,
        epochs=30,
        device=device,
        seed=seed + 5000,
    )

    # (D) Small-only ERM on S1
    model_small_only = train_erm_on_labeled(
        X_1,
        Y1_std,
        dim=dim,
        hidden=64,
        epochs=80,
        device=device,
        seed=seed + 6000,
    )

    # Evaluate on full population (standardized)
    mse_cal_active = eval_mse(model_cal_active, X, Y_std, device=device)
    mse_raw_active = eval_mse(model_raw_active, X, Y_std, device=device)
    mse_cal_random = eval_mse(model_cal_random, X, Y_std, device=device)
    mse_small_only = eval_mse(model_small_only, X, Y_std, device=device)

    return {
        "mse_cal_active": mse_cal_active,
        "mse_raw_active": mse_raw_active,
        "mse_cal_random": mse_cal_random,
        "mse_small_only": mse_small_only,
        "el_info": info,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N, dim = 10000, 10
    n_1, n_2 = 100, 1000
    budget = 500
    n_sim = 200
    is_linear = False
    use_fixed_budget = False  # Poisson is theory-clean for xi/pi

    out = {"cal_active": [], "raw_active": [], "cal_random": [], "small_only": []}

    for s in tqdm(range(n_sim)):
        r = simulate_once(
            N=N,
            dim=dim,
            n_1=n_1,
            n_2=n_2,
            budget=budget,
            is_linear=is_linear,
            seed=s,
            device=device,
            use_fixed_budget=use_fixed_budget,
            K=5,
        )
        out["cal_active"].append(r["mse_cal_active"])
        out["raw_active"].append(r["mse_raw_active"])
        out["cal_random"].append(r["mse_cal_random"])
        out["small_only"].append(r["mse_small_only"])

    for k in out:
        out[k] = np.array(out[k])

    print("=== M-estimation (FCNN) comparison (standardized Y) ===")
    print(f"Small-only (S1 ERM)          : mean {out['small_only'].mean():.4f} (std {out['small_only'].std():.4f})")
    print(f"Calibrated Random (S2 rand)  : mean {out['cal_random'].mean():.4f} (std {out['cal_random'].std():.4f})")
    print(f"Raw Active (u only)          : mean {out['raw_active'].mean():.4f} (std {out['raw_active'].std():.4f})")
    print(f"Calibrated Active (p*u)      : mean {out['cal_active'].mean():.4f} (std {out['cal_active'].std():.4f})")
    print(f"Gain vs Cal-Random (rand - calActive): {(out['cal_random'] - out['cal_active']).mean():.4f}")
