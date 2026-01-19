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


def fit_models(X_1, Y_1, random_state=None):
    """Fit regression and uncertainty models using XGBoost."""
    model_reg = XGBRegressor(booster='gbtree', tree_method='hist', n_jobs=128, random_state=random_state)
    model_reg.fit(X_1, Y_1)
    Y_1_pred = model_reg.predict(X_1)
    res_abs = np.abs(Y_1 - Y_1_pred)
    model_unc = XGBRegressor(booster='gbtree', tree_method='hist', n_jobs=128, random_state=random_state)
    model_unc.fit(X_1, res_abs)
    return model_reg, model_unc

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    seed_everything(seed)
    
    # generate data
    N = 1000
    dim = 10
    n_1 = 100
    n_2 =  500
    sample_budget = 200
    n_sim = 1000
    is_linear = False
    use_mu_x_from_s1 = False
    

    def simulate_one_time(sim_seed=None):
        mu_X, mu_Y, X_1, Y_1, Y_2, X_2 = generate_finite_population(
            N, dim, n_1, n_2, is_linear=is_linear, use_quadratic_features=not is_linear, seed=sim_seed
        )
        model_reg, model_unc = fit_models(X_1, Y_1, random_state=sim_seed)
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
        # Use different seed for each simulation to ensure reproducibility while allowing variation
        sim_seed = seed + i if seed is not None else None
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
        ) = simulate_one_time(sim_seed=sim_seed)
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

    # Compute coverage rates
    coverage_calibrated = compute_coverage(
        lst_activate_estimator_calibrated, lst_mu_Y, lst_var_calibrated
    )
    coverage_cal_s1 = compute_coverage(
        lst_activate_estimator_calibrated_s1, lst_mu_Y, lst_var_calibrated_s1
    )
    coverage_raw = compute_coverage(lst_activate_estimator_raw, lst_mu_Y, lst_var_raw)
    coverage_cal_random = compute_coverage(
        lst_activate_estimator_cal_random, lst_mu_Y, lst_var_cal_random
    )
    coverage_small_only = compute_coverage(lst_small_only, lst_mu_Y, lst_var_small_only)

    # Create and display results table
    methods = [
        "Calibrated Active (mu_X true)",
        "Calibrated Active (mu_X S1)",
        "Raw Active",
        "Calibrated Random",
        "Small-only (S1 mean)",
    ]
    biases = [bias_calibrated, bias_cal_s1, bias_raw, bias_cal_random, bias_small]
    variances = [variance_calibrated, variance_cal_s1, variance_raw, variance_cal_random, variance_small]
    mses = [mse_calibrated, mse_cal_s1, mse_raw, mse_cal_random, mse_small]
    coverages = [coverage_calibrated, coverage_cal_s1, coverage_raw, coverage_cal_random, coverage_small_only]

    table = create_results_table(methods, biases, variances, mses, coverages)
    console = Console()
    console.print(table)


