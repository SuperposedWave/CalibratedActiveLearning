import numpy as np
import random
from xgboost import XGBRegressor
from calibrated_active_learning import estimate_p, estimate_pi, sample_by_pi, get_activate_estimator
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_finite_population(N, dim, n_1, n_2, is_linear=True):
    # generate N points in dim-dimensional space
    X = np.random.randn(N, dim)
    noise = np.random.randn(N)

    if is_linear:
        beta = np.random.uniform(-5, 5, dim)
        Y = X @ beta + noise + 100
        mu_X = np.mean(X, axis=0)
    else:
        prod = X[:,:,None] * X[:,None,:]
        prod = prod.reshape(N, dim*dim)
        # stack prod and X
        X = np.concatenate([X, prod], axis=1)
        beta = np.random.uniform(-5, 5, X.shape[1])
        Y = X @ beta + noise + 100
        # 在扩展 X 之后计算 mu_X
        mu_X = np.mean(X, axis=0)
    
    mu_Y = np.mean(Y)
    # sample a small sample S_1 from X by simple random sampling
    S_1 = np.random.choice(N, size=n_1, replace=False)
    X_1 = X[S_1]
    Y_1 = Y[S_1]

    # sample a large sample S_2 from X by a biased sampling design
    score = 1.5 * X[:, 0] + 0.5 * X[:, 1]  # 只依赖 covariates
    p = 1 / (1 + np.exp(-score))           # logistic
    p = p / p.sum()
    S_2 = np.random.choice(N, size=n_2, replace=False, p=p)
    X_2 = X[S_2]
    Y_2 = Y[S_2]


    return mu_X, mu_Y, X_1, Y_1, Y_2,X_2


def fit_models(X_1, Y_1):
    model_reg = XGBRegressor()
    model_reg.fit(X_1, Y_1)
    Y_1_pred = model_reg.predict(X_1)
    res_abs = np.abs(Y_1 - Y_1_pred)
    model_unc = XGBRegressor()
    model_unc.fit(X_1, res_abs)
    return model_reg, model_unc


def summarize_metrics(estimates, truth):
    bias = np.mean(estimates - truth)
    variance = np.var(estimates - truth)
    mse = np.mean((estimates - truth) ** 2)
    return bias, variance, mse

if __name__ == "__main__":
    # generate data
    N = 10000
    dim = 10
    n_1 = 500
    n_2 =  5000
    sample_budget = 1000
    n_sim = 100
    is_linear = False
    use_mu_x_from_s1 = False
    

    def simulate_one_time():
        mu_X, mu_Y, X_1, Y_1, Y_2, X_2 = generate_finite_population(N, dim, n_1, n_2, is_linear)
        model_reg, model_unc = fit_models(X_1, Y_1)
        mu_X_s1 = np.mean(X_1, axis=0)
        # Use estimate_p from package
        p = estimate_p(X_2, mu_X)
        p_s1 = estimate_p(X_2, mu_X_s1)
        y_pred = model_reg.predict(X_2)
        # uncertainty = model_unc.predict(X_2)
        uncertainty = np.maximum(model_unc.predict(X_2), 0.0) + 1e-12

        # Use estimate_pi and sample_by_pi from package
        pi_calibrated = estimate_pi(uncertainty, sample_budget, p)
        xi_calibrated = sample_by_pi(pi_calibrated)

        activate_estimator, var_calibrated = get_activate_estimator(
            y_pred, Y_2, pi_calibrated, xi_calibrated, p=p, estimate_variance=True
        )

        # Raw active: pi based on uncertainty only, uniform weights
        pi_raw = estimate_pi(uncertainty, sample_budget, p=None)
        xi_raw = sample_by_pi(pi_raw)
        p_raw = np.ones(n_2) / n_2
        activate_estimator_raw, var_raw = get_activate_estimator(
            y_pred, Y_2, pi_raw, xi_raw, p=p_raw, estimate_variance=True
        )

        # Calibrated active with S1 mu_X
        pi_calibrated_s1 = estimate_pi(uncertainty, sample_budget, p_s1)
        xi_calibrated_s1 = sample_by_pi(pi_calibrated_s1)
        activate_estimator_s1, var_calibrated_s1 = get_activate_estimator(
            y_pred, Y_2, pi_calibrated_s1, xi_calibrated_s1, p=p_s1, estimate_variance=True
        )

        # Calibrated random
        pi_rand = np.ones(n_2) * (sample_budget / n_2)
        pi_rand = np.clip(pi_rand, 1e-12, 1 - 1e-12)
        xi_rand = sample_by_pi(pi_rand)
        activate_estimator_cal_random, var_cal_random = get_activate_estimator(
            y_pred, Y_2, pi_rand, xi_rand, p=p, estimate_variance=True
        )
        small_only = np.mean(Y_1)
        var_small_only = np.var(Y_1, ddof=1) / n_1
        return (
            mu_Y,
            activate_estimator,
            activate_estimator_raw,
            activate_estimator_cal_random,
            activate_estimator_s1,
            small_only,
            var_calibrated,
            var_raw,
            var_cal_random,
            var_calibrated_s1,
            var_small_only,
        )

    # print("mu_Y_raw =", np.mean(Y_2))
    lst_mu_Y = []
    lst_activate_estimator_calibrated = []
    lst_activate_estimator_raw = []
    lst_activate_estimator_cal_random = []
    lst_activate_estimator_calibrated_s1 = []
    lst_small_only = []
    lst_var_calibrated = []
    lst_var_raw = []
    lst_var_cal_random = []
    lst_var_calibrated_s1 = []
    lst_var_small_only = []
    for i in tqdm(range(n_sim)):
        (
            mu_Y,
            activate_estimator,
            activate_estimator_raw,
            activate_estimator_cal_random,
            activate_estimator_s1,
            small_only,
            var_calibrated,
            var_raw,
            var_cal_random,
            var_calibrated_s1,
            var_small_only,
        ) = simulate_one_time()
        lst_mu_Y.append(mu_Y)
        lst_activate_estimator_calibrated.append(activate_estimator)
        lst_activate_estimator_raw.append(activate_estimator_raw)
        lst_activate_estimator_cal_random.append(activate_estimator_cal_random)
        lst_activate_estimator_calibrated_s1.append(activate_estimator_s1)
        lst_small_only.append(small_only)
        lst_var_calibrated.append(var_calibrated)
        lst_var_raw.append(var_raw)
        lst_var_cal_random.append(var_cal_random)
        lst_var_calibrated_s1.append(var_calibrated_s1)
        lst_var_small_only.append(var_small_only)
    # bias
    lst_activate_estimator_calibrated = np.array(lst_activate_estimator_calibrated)
    lst_activate_estimator_raw = np.array(lst_activate_estimator_raw)
    lst_activate_estimator_cal_random = np.array(lst_activate_estimator_cal_random)
    lst_activate_estimator_calibrated_s1 = np.array(lst_activate_estimator_calibrated_s1)
    lst_small_only = np.array(lst_small_only)
    lst_mu_Y = np.array(lst_mu_Y)
    lst_var_calibrated = np.array(lst_var_calibrated)
    lst_var_raw = np.array(lst_var_raw)
    lst_var_cal_random = np.array(lst_var_cal_random)
    lst_var_calibrated_s1 = np.array(lst_var_calibrated_s1)
    lst_var_small_only = np.array(lst_var_small_only)
    bias_calibrated, variance_calibrated, mse_calibrated = summarize_metrics(
        lst_activate_estimator_calibrated, lst_mu_Y
    )
    bias_raw, variance_raw, mse_raw = summarize_metrics(lst_activate_estimator_raw, lst_mu_Y)
    bias_cal_random, variance_cal_random, mse_cal_random = summarize_metrics(
        lst_activate_estimator_cal_random, lst_mu_Y
    )
    bias_cal_s1, variance_cal_s1, mse_cal_s1 = summarize_metrics(
        lst_activate_estimator_calibrated_s1, lst_mu_Y
    )
    bias_small, variance_small, mse_small = summarize_metrics(lst_small_only, lst_mu_Y)

    z_map = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    coverage_calibrated = {}
    coverage_cal_s1 = {}
    coverage_raw = {}
    coverage_cal_random = {}
    coverage_small_only = {}
    for level, z in z_map.items():
        coverage_calibrated[level] = np.mean(
            (lst_mu_Y >= lst_activate_estimator_calibrated - z * np.sqrt(lst_var_calibrated))
            & (lst_mu_Y <= lst_activate_estimator_calibrated + z * np.sqrt(lst_var_calibrated))
        )
        coverage_cal_s1[level] = np.mean(
            (lst_mu_Y >= lst_activate_estimator_calibrated_s1 - z * np.sqrt(lst_var_calibrated_s1))
            & (lst_mu_Y <= lst_activate_estimator_calibrated_s1 + z * np.sqrt(lst_var_calibrated_s1))
        )
        coverage_raw[level] = np.mean(
            (lst_mu_Y >= lst_activate_estimator_raw - z * np.sqrt(lst_var_raw))
            & (lst_mu_Y <= lst_activate_estimator_raw + z * np.sqrt(lst_var_raw))
        )
        coverage_cal_random[level] = np.mean(
            (lst_mu_Y >= lst_activate_estimator_cal_random - z * np.sqrt(lst_var_cal_random))
            & (lst_mu_Y <= lst_activate_estimator_cal_random + z * np.sqrt(lst_var_cal_random))
        )
        coverage_small_only[level] = np.mean(
            (lst_mu_Y >= lst_small_only - z * np.sqrt(lst_var_small_only))
            & (lst_mu_Y <= lst_small_only + z * np.sqrt(lst_var_small_only))
        )

    table = Table(title="Active Estimator Comparison")
    table.add_column("Method", style="bold")
    table.add_column("Bias", justify="right")
    table.add_column("Variance", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("Coverage(90%)", justify="right")
    table.add_column("Coverage(95%)", justify="right")
    table.add_column("Coverage(99%)", justify="right")
    table.add_row(
        "Calibrated Active (mu_X true)",
        f"{bias_calibrated:.6f}",
        f"{variance_calibrated:.6f}",
        f"{mse_calibrated:.6f}",
        f"{coverage_calibrated[0.90]:.3f}",
        f"{coverage_calibrated[0.95]:.3f}",
        f"{coverage_calibrated[0.99]:.3f}",
    )
    table.add_row(
        "Calibrated Active (mu_X S1)",
        f"{bias_cal_s1:.6f}",
        f"{variance_cal_s1:.6f}",
        f"{mse_cal_s1:.6f}",
        f"{coverage_cal_s1[0.90]:.3f}",
        f"{coverage_cal_s1[0.95]:.3f}",
        f"{coverage_cal_s1[0.99]:.3f}",
    )
    table.add_row(
        "Raw Active",
        f"{bias_raw:.6f}",
        f"{variance_raw:.6f}",
        f"{mse_raw:.6f}",
        f"{coverage_raw[0.90]:.3f}",
        f"{coverage_raw[0.95]:.3f}",
        f"{coverage_raw[0.99]:.3f}",
    )
    table.add_row(
        "Calibrated Random",
        f"{bias_cal_random:.6f}",
        f"{variance_cal_random:.6f}",
        f"{mse_cal_random:.6f}",
        f"{coverage_cal_random[0.90]:.3f}",
        f"{coverage_cal_random[0.95]:.3f}",
        f"{coverage_cal_random[0.99]:.3f}",
    )
    table.add_row(
        "Small-only (S1 mean)",
        f"{bias_small:.6f}",
        f"{variance_small:.6f}",
        f"{mse_small:.6f}",
        f"{coverage_small_only[0.90]:.3f}",
        f"{coverage_small_only[0.95]:.3f}",
        f"{coverage_small_only[0.99]:.3f}",
    )
    console = Console()
    console.print(table)


