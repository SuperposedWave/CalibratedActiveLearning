import numpy as np
from xgboost import XGBRegressor
from calibrated_active_learning import estimate_p, estimate_pi, sample_by_pi, get_activate_estimator
from tqdm import tqdm
from rich.console import Console
from utils import (
    seed_everything,
    generate_finite_population,
    summarize_metrics,
    compute_coverage,
    create_results_table,
)


def fit_models(X_1, Y_1, random_state=None, n_splits=5):
    """Fit regression and uncertainty models using cross-fitted residuals."""
    model_reg = XGBRegressor(
        booster='gbtree',
        tree_method='hist',
        n_jobs=128,
        random_state=random_state,
        n_estimators=200,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model_unc = XGBRegressor(
        booster='gbtree',
        tree_method='hist',
        n_jobs=128,
        random_state=random_state,
        n_estimators=200,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    n = X_1.shape[0]
    splits = max(2, min(n_splits, n))
    if n < 2:
        model_reg.fit(X_1, Y_1)
        res_abs = np.abs(Y_1 - model_reg.predict(X_1))
        model_unc.fit(X_1, res_abs)
        return model_reg, model_unc

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    fold_sizes = np.full(splits, n // splits, dtype=int)
    fold_sizes[: n % splits] += 1
    oof_resid = np.empty(n, dtype=float)
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        val_idx = perm[start:stop]
        train_idx = np.concatenate([perm[:start], perm[stop:]])
        fold_reg = XGBRegressor(
            booster='gbtree',
            tree_method='hist',
            n_jobs=128,
            random_state=random_state,
            n_estimators=200,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        fold_reg.fit(X_1[train_idx], Y_1[train_idx])
        y_val_pred = fold_reg.predict(X_1[val_idx])
        oof_resid[val_idx] = np.abs(Y_1[val_idx] - y_val_pred)
        start = stop

    model_reg.fit(X_1, Y_1)
    model_unc.fit(X_1, oof_resid)
    return model_reg, model_unc

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    seed_everything(seed)
    
    # generate data
    N = 1000
    dim = 5
    n_1 = 50
    n_2 =  500
    sample_budget = 200
    n_sim = 100
    is_linear = True
    use_mu_x_from_s1 = False
    tau_mix = 0.5
    

    def simulate_one_time(sim_seed=None):
        mu_X, mu_Y, X_1, Y_1, Y_2, X_2 = generate_finite_population(
            N, dim, n_1, n_2, is_linear=is_linear, use_quadratic_features=not is_linear, seed=sim_seed
        )
        model_reg, model_unc = fit_models(X_1, Y_1, random_state=sim_seed)
        mu_X_s1 = np.mean(X_1, axis=0)
        # Use estimate_p from package
        p = estimate_p(X_2, mu_X)
        p_s1 = estimate_p(X_2, mu_X_s1)
        mu_x_s1_diff = mu_X_s1 - mu_X
        mu_x_s1_diff_norm = np.linalg.norm(mu_x_s1_diff)
        moment_error = (p[:, None] * X_2).sum(axis=0) - mu_X
        moment_error_norm = np.linalg.norm(moment_error)
        p_max = np.max(p)
        p_min = np.min(p)
        y_pred = model_reg.predict(X_2)
        uncertainty = np.maximum(model_unc.predict(X_2), 1e-12)

        # Use estimate_pi and sample_by_pi from package
        pi_calibrated = estimate_pi(uncertainty, sample_budget, p, tau=tau_mix)
        xi_calibrated = sample_by_pi(pi_calibrated)

        activate_estimator, var_calibrated = get_activate_estimator(
            y_pred, Y_2, pi_calibrated, xi_calibrated, p=p, estimate_variance=True
        )

        # Raw active: pi based on uncertainty only, uniform weights 
        pi_raw = estimate_pi(uncertainty, sample_budget, p=None, tau=tau_mix)
        xi_raw = sample_by_pi(pi_raw)
        p_raw = np.ones(n_2) / n_2
        activate_estimator_raw, var_raw = get_activate_estimator(
            y_pred, Y_2, pi_raw, xi_raw, p=p_raw, estimate_variance=True
        )

        # Calibrated EL-only: pi based on EL weights only (no uncertainty)
        pi_el = estimate_pi(np.ones(n_2), sample_budget, p, tau=tau_mix)
        xi_el = sample_by_pi(pi_el)
        activate_estimator_el, var_el = get_activate_estimator(
            y_pred, Y_2, pi_el, xi_el, p=p, estimate_variance=True
        )

        # Calibrated active with S1 mu_X
        pi_calibrated_s1 = estimate_pi(uncertainty, sample_budget, p_s1, tau=tau_mix)
        xi_calibrated_s1 = sample_by_pi(pi_calibrated_s1)
        activate_estimator_s1, var_calibrated_s1 = get_activate_estimator(
            y_pred, Y_2, pi_calibrated_s1, xi_calibrated_s1, p=p_s1, estimate_variance=True
        )

        # Calibrated random
        pi_rand = np.ones(n_2) * (sample_budget / n_2)
        xi_rand = sample_by_pi(pi_rand)
        activate_estimator_cal_random, var_cal_random = get_activate_estimator(
            y_pred, Y_2, pi_rand, xi_rand, p=p, estimate_variance=True
        )
        small_only = np.mean(Y_1)
        var_small_only = np.var(Y_1, ddof=1) / n_1
        p_weighted_mu_y = np.sum(p * Y_2)
        p_weighted_bias = p_weighted_mu_y - mu_Y
        p_s1_weighted_mu_y = np.sum(p_s1 * Y_2)
        p_s1_weighted_bias = p_s1_weighted_mu_y - mu_Y
        unweighted_bias = np.mean(Y_2) - mu_Y
        return (
            mu_Y,
            activate_estimator,
            activate_estimator_raw,
            activate_estimator_el,
            activate_estimator_cal_random,
            activate_estimator_s1,
            small_only,
            var_calibrated,
            var_raw,
            var_el,
            var_cal_random,
            var_calibrated_s1,
            var_small_only,
            p_weighted_bias,
            p_s1_weighted_bias,
            unweighted_bias,
            moment_error_norm,
            p_max,
            p_min,
            mu_x_s1_diff_norm,
        )

    # print("mu_Y_raw =", np.mean(Y_2))
    lst_mu_Y = []
    lst_activate_estimator_calibrated = []
    lst_activate_estimator_raw = []
    lst_activate_estimator_el = []
    lst_activate_estimator_cal_random = []
    lst_activate_estimator_calibrated_s1 = []
    lst_small_only = []
    lst_var_calibrated = []
    lst_var_raw = []
    lst_var_el = []
    lst_var_cal_random = []
    lst_var_calibrated_s1 = []
    lst_var_small_only = []
    lst_p_weighted_bias = []
    lst_p_s1_weighted_bias = []
    lst_unweighted_bias = []
    lst_moment_error_norm = []
    lst_p_max = []
    lst_p_min = []
    lst_mu_x_s1_diff_norm = []
    for i in tqdm(range(n_sim)):
        # Use different seed for each simulation to ensure reproducibility while allowing variation
        sim_seed = seed + i if seed is not None else None
        (
            mu_Y,
            activate_estimator,
            activate_estimator_raw,
            activate_estimator_el,
            activate_estimator_cal_random,
            activate_estimator_s1,
            small_only,
            var_calibrated,
            var_raw,
            var_el,
            var_cal_random,
            var_calibrated_s1,
            var_small_only,
            p_weighted_bias,
            p_s1_weighted_bias,
            unweighted_bias,
            moment_error_norm,
            p_max,
            p_min,
            mu_x_s1_diff_norm,
        ) = simulate_one_time(sim_seed=sim_seed)
        lst_mu_Y.append(mu_Y)
        lst_activate_estimator_calibrated.append(activate_estimator)
        lst_activate_estimator_raw.append(activate_estimator_raw)
        lst_activate_estimator_el.append(activate_estimator_el)
        lst_activate_estimator_cal_random.append(activate_estimator_cal_random)
        lst_activate_estimator_calibrated_s1.append(activate_estimator_s1)
        lst_small_only.append(small_only)
        lst_var_calibrated.append(var_calibrated)
        lst_var_raw.append(var_raw)
        lst_var_el.append(var_el)
        lst_var_cal_random.append(var_cal_random)
        lst_var_calibrated_s1.append(var_calibrated_s1)
        lst_var_small_only.append(var_small_only)
        lst_p_weighted_bias.append(p_weighted_bias)
        lst_p_s1_weighted_bias.append(p_s1_weighted_bias)
        lst_unweighted_bias.append(unweighted_bias)
        lst_moment_error_norm.append(moment_error_norm)
        lst_p_max.append(p_max)
        lst_p_min.append(p_min)
        lst_mu_x_s1_diff_norm.append(mu_x_s1_diff_norm)
    # bias
    lst_activate_estimator_calibrated = np.array(lst_activate_estimator_calibrated)
    lst_activate_estimator_raw = np.array(lst_activate_estimator_raw)
    lst_activate_estimator_el = np.array(lst_activate_estimator_el)
    lst_activate_estimator_cal_random = np.array(lst_activate_estimator_cal_random)
    lst_activate_estimator_calibrated_s1 = np.array(lst_activate_estimator_calibrated_s1)
    lst_small_only = np.array(lst_small_only)
    lst_mu_Y = np.array(lst_mu_Y)
    lst_var_calibrated = np.array(lst_var_calibrated)
    lst_var_raw = np.array(lst_var_raw)
    lst_var_el = np.array(lst_var_el)
    lst_var_cal_random = np.array(lst_var_cal_random)
    lst_var_calibrated_s1 = np.array(lst_var_calibrated_s1)
    lst_var_small_only = np.array(lst_var_small_only)
    bias_calibrated, variance_calibrated, mse_calibrated = summarize_metrics(
        lst_activate_estimator_calibrated, lst_mu_Y
    )
    bias_raw, variance_raw, mse_raw = summarize_metrics(lst_activate_estimator_raw, lst_mu_Y)
    bias_el, variance_el, mse_el = summarize_metrics(lst_activate_estimator_el, lst_mu_Y)
    bias_cal_random, variance_cal_random, mse_cal_random = summarize_metrics(
        lst_activate_estimator_cal_random, lst_mu_Y
    )
    bias_cal_s1, variance_cal_s1, mse_cal_s1 = summarize_metrics(
        lst_activate_estimator_calibrated_s1, lst_mu_Y
    )
    bias_small, variance_small, mse_small = summarize_metrics(lst_small_only, lst_mu_Y)

    # Compute coverage rates
    coverage_calibrated = compute_coverage(
        lst_activate_estimator_calibrated, lst_mu_Y, lst_var_calibrated
    )
    coverage_cal_s1 = compute_coverage(
        lst_activate_estimator_calibrated_s1, lst_mu_Y, lst_var_calibrated_s1
    )
    coverage_raw = compute_coverage(lst_activate_estimator_raw, lst_mu_Y, lst_var_raw)
    coverage_el = compute_coverage(lst_activate_estimator_el, lst_mu_Y, lst_var_el)
    coverage_cal_random = compute_coverage(
        lst_activate_estimator_cal_random, lst_mu_Y, lst_var_cal_random
    )
    coverage_small_only = compute_coverage(lst_small_only, lst_mu_Y, lst_var_small_only)

    # Create and display results table
    methods = [
        "Calibrated Active (mu_X true)",
        "Calibrated Active (mu_X S1)",
        "Raw Active",
        "Calibrated EL-only",
        "Calibrated Random",
        "Small-only (S1 mean)",
    ]
    biases = [bias_calibrated, bias_cal_s1, bias_raw, bias_el, bias_cal_random, bias_small]
    variances = [variance_calibrated, variance_cal_s1, variance_raw, variance_el, variance_cal_random, variance_small]
    mses = [mse_calibrated, mse_cal_s1, mse_raw, mse_el, mse_cal_random, mse_small]
    coverages = [
        coverage_calibrated,
        coverage_cal_s1,
        coverage_raw,
        coverage_el,
        coverage_cal_random,
        coverage_small_only,
    ]

    table = create_results_table(methods, biases, variances, mses, coverages)
    console = Console()
    console.print(table)

    p_weighted_bias = np.array(lst_p_weighted_bias)
    p_s1_weighted_bias = np.array(lst_p_s1_weighted_bias)
    unweighted_bias = np.array(lst_unweighted_bias)
    print(
        f"\nEL加权 Y 均值偏差（均值 ± 标准差）: {p_weighted_bias.mean():.6f} ± {p_weighted_bias.std():.6f}"
    )
    print(
        f"EL(S1均值)加权 Y 均值偏差（均值 ± 标准差）: {p_s1_weighted_bias.mean():.6f} ± {p_s1_weighted_bias.std():.6f}"
    )
    print(
        f"未加权 Y_2 均值偏差（均值 ± 标准差）: {unweighted_bias.mean():.6f} ± {unweighted_bias.std():.6f}"
    )
    moment_error_norm = np.array(lst_moment_error_norm)
    p_max = np.array(lst_p_max)
    p_min = np.array(lst_p_min)
    print(
        f"EL矩条件误差范数（均值 ± 标准差）: {moment_error_norm.mean():.6f} ± {moment_error_norm.std():.6f}"
    )
    print(f"EL权重范围（均值）: min={p_min.mean():.6e}, max={p_max.mean():.6e}")
    mu_x_s1_diff_norm = np.array(lst_mu_x_s1_diff_norm)
    print(
        f"mu_X(S1) 与 mu_X 差异范数（均值 ± 标准差）: {mu_x_s1_diff_norm.mean():.6f} ± {mu_x_s1_diff_norm.std():.6f}"
    )
