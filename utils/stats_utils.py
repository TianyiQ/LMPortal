import math
import os
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.metrics import log_loss
from statsmodels.stats.outliers_influence import OLSInfluence
from tqdm import tqdm


def fisher_r_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return (float("nan"), float("nan"))
    # clamp r to avoid atanh overflow
    r = max(min(r, 0.999999), -0.999999)
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))


def scatter_with_outliers_and_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    include_stats_in_title: bool = True,
    mark_outliers: bool = None,
    cook_threshold: Optional[float] = None,
    ci_alpha: float = 0.05,
    point_kwargs: Optional[dict[str, Any]] = None,
    outlier_kwargs: Optional[dict[str, Any]] = None,
    line_kwargs: Optional[dict[str, Any]] = None,
    ci_kwargs: Optional[dict[str, Any]] = None,
    supress_reference_line: bool = False,
    force_linear_scale: bool = False,
    x_ci_range: Optional[list[Optional[tuple[float, float]]]] = None,
    y_ci_range: Optional[list[Optional[tuple[float, float]]]] = None,
    group_ids: Optional[list[Any]] = None,
    x_scale: Literal["auto", "linear", "log"] = "auto",
    y_scale: Literal["auto", "linear", "log"] = "auto",
) -> dict[str, Any]:
    """Render a scatter plot with OLS fit, CI band, outlier marking, and Pearson rho with CI.
    :param x: x values
    :type x: np.ndarray
    :param y: y values
    :type y: np.ndarray
    :param ax: matplotlib axes to plot on
    :type ax: plt.Axes, optional
    :param xlabel: x label
    :type xlabel: str, optional
    :param ylabel: y label
    :type ylabel: str, optional
    :param title: title
    :type title: str, optional
    :param include_stats_in_title: include stats in title
    :type include_stats_in_title: bool, optional
    :param mark_outliers: mark outliers
    :type mark_outliers: bool, optional
    :param cook_threshold: cook's distance threshold
    :type cook_threshold: float, optional
    :param ci_alpha: confidence interval alpha
    :type ci_alpha: float, optional
    :param point_kwargs: kwargs for points
    :type point_kwargs: dict[str, Any], optional
    :param outlier_kwargs: kwargs for outliers
    :type outlier_kwargs: dict[str, Any], optional
    :param line_kwargs: kwargs for line
    :type line_kwargs: dict[str, Any], optional
    :param ci_kwargs: kwargs for CI band
    :type ci_kwargs: dict[str, Any], optional
    :param supress_reference_line: suppress reference line
    :type supress_reference_line: bool, optional
    :param x_ci_range: x CI range
    :type x_ci_range: list[Optional[tuple[float, float]]], optional
    :param y_ci_range: y CI range
    :type y_ci_range: list[Optional[tuple[float, float]]], optional
    :param group_ids: optionally group points by a categorical variable, which will be used to color the points & do regression separately for each group
    :type group_ids: list[Any], optional
    :param x_scale: x scale
    :type x_scale: Literal["auto", "linear", "log"], optional
    :param y_scale: y scale
    :type y_scale: Literal["auto", "linear", "log"], optional
    Returns a dict of computed statistics and masks.
    """
    if group_ids is None:
        group_ids = ["ALL"] * len(x)

    x_log_scale = x_scale == "log"
    y_log_scale = y_scale == "log"
    if x_scale == "auto" and (np.mean(x) > 2 * np.median(x) or np.max(x) > 6 * np.median(x)) and np.min(x) > 0:
        x_log_scale = True
    if y_scale == "auto" and (np.mean(y) > 2 * np.median(y) or np.max(y) > 6 * np.median(y)) and np.min(y) > 0:
        y_log_scale = True

    if int(os.getenv("DEBUG", "0")):
        print(f"x_log_scale: {x_log_scale}, y_log_scale: {y_log_scale}")

    if mark_outliers is None:
        mark_outliers = len(x) > 10 * len(set(group_ids))

    unique_group_ids = sorted(list(set(group_ids)))
    group_ids = np.array(group_ids)
    group_color_candidates = plt.get_cmap("tab10").colors
    group_colors = {
        group_id: group_color_candidates[i % len(group_color_candidates)] for i, group_id in enumerate(unique_group_ids)
    }

    plt.clf()
    plt.figure(figsize=(10, 8))
    if ax is None:
        ax = plt.gca()

    all_x = np.asarray(x, dtype=float)
    all_y = np.asarray(y, dtype=float)
    if x_ci_range:
        all_x_ci_range = np.asarray(x_ci_range, dtype=float).T
        all_x_ci_range = np.abs(all_x_ci_range - all_x.reshape(1, -1))  # Convert from (lo, hi) to (x-lo, hi-x)
    if y_ci_range:
        all_y_ci_range = np.asarray(y_ci_range, dtype=float).T
        all_y_ci_range = np.abs(all_y_ci_range - all_y.reshape(1, -1))  # Convert from (lo, hi) to (x-lo, hi-x)

    lines = [title] if title else []

    for group_id in unique_group_ids:
        group_mask = group_ids == group_id
        x = all_x[group_mask]
        y = all_y[group_mask]
        x_ci_range = all_x_ci_range[:, group_mask] if x_ci_range is not None else None
        y_ci_range = all_y_ci_range[:, group_mask] if y_ci_range is not None else None
        inlier_color = group_colors[group_id] if len(unique_group_ids) > 1 else "#1f77b4"
        outlier_color = inlier_color if len(unique_group_ids) > 1 else "red"
        regression_color = inlier_color if len(unique_group_ids) > 1 else "red"

        n = len(x)
        if n != len(y) or n < 2:
            if int(os.getenv("DEBUG", "0")):
                print(f"x and y must have same length and at least 2 points: {n} != {len(y)} or {n} < 2")
                print(f"group_id: {group_id}")
                print(f"x: {x}")
                print(f"y: {y}")
            raise ValueError("x and y must have same length and at least 2 points")

        # OLS fit y ~ const + x
        reg_x = np.log(x) if x_log_scale else x
        reg_y = np.log(y) if y_log_scale else y
        X = sm.add_constant(reg_x)
        ols = sm.OLS(reg_y, X).fit()
        coef = float(ols.params[1]) if len(ols.params) > 1 else float("nan")
        intercept = float(ols.params[0])

        def _predict_y(x: np.ndarray, coef: float, intercept: float) -> np.ndarray:
            if x_log_scale:
                x = np.log(x)

            predicted_y = coef * x + intercept
            if y_log_scale:
                predicted_y = np.exp(predicted_y)
            return predicted_y

        # Outliers by Cook's distance
        infl = OLSInfluence(ols)
        cooks = infl.cooks_distance[0]
        thr = cook_threshold if cook_threshold is not None else (4.0 / n)
        if not mark_outliers:
            thr = float("inf")
        is_out = cooks > thr

        # Scatter: inliers and outliers
        pk = {"s": 30, "alpha": 0.6, "color": inlier_color}
        if len(unique_group_ids) > 1:
            pk["label"] = f"Group {group_id}"
        if point_kwargs:
            pk.update(point_kwargs)
        inliers = ~is_out
        ax.scatter(x[inliers], y[inliers], **pk)

        # Show error bars for x and y
        if x_ci_range is not None or y_ci_range is not None:
            ax.errorbar(
                x[inliers],
                y[inliers],
                xerr=(x_ci_range[:, inliers] if x_ci_range is not None else None),
                yerr=(y_ci_range[:, inliers] if y_ci_range is not None else None),
                fmt="o",
                color=inlier_color,
                alpha=0.5,
            )

        # Regression line across sorted x
        order = np.argsort(x)
        x_sorted = x[order]
        y_pred_sorted = _predict_y(x_sorted, coef, intercept)
        # Copy design: red dashed regression
        lk = {"color": regression_color, "linestyle": "--", "alpha": 0.8}
        if len(unique_group_ids) > 1:
            lk["label"] = f"Regression (Group {group_id})"
        if line_kwargs:
            lk.update(line_kwargs)
        ax.plot(x_sorted, y_pred_sorted, **lk)

        # 95% CI band around regression line via prediction interval on mean
        reg_x_line = np.linspace(reg_x.min(), reg_x.max(), 200)
        reg_X_line = sm.add_constant(reg_x_line)
        pred_line = ols.get_prediction(reg_X_line)
        ci = pred_line.conf_int(alpha=ci_alpha)
        ck = {"color": regression_color, "alpha": 0.15}
        if ci_kwargs:
            ck.update(ci_kwargs)
        ax.fill_between(
            np.exp(reg_x_line) if x_log_scale else reg_x_line,
            np.exp(ci[:, 0]) if y_log_scale else ci[:, 0],
            np.exp(ci[:, 1]) if y_log_scale else ci[:, 1],
            **ck,
        )

        # Diagonal line for perfect agreement
        min_val = float(min(x.min(), y.min()))
        max_val = float(max(x.max(), y.max()))
        if not supress_reference_line:
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect Agreement")

        # Plot outliers last so they are on top of all elements
        if is_out.any():
            ok = {
                "s": 80,
                "color": outlier_color,
                "alpha": 0.9,
                "edgecolors": "black",
                "linewidth": 1.2,
                "label": f"Outliers (n={int(is_out.sum())}"
                + (f" for Group {group_id})" if len(unique_group_ids) > 1 else ")"),
                "zorder": 10,
            }
            if outlier_kwargs:
                ok.update(outlier_kwargs)
            ax.scatter(x[is_out], y[is_out], **ok)
            if x_ci_range is not None or y_ci_range is not None:
                ax.errorbar(
                    x[is_out],
                    y[is_out],
                    xerr=(x_ci_range[:, is_out] if x_ci_range is not None else None),
                    yerr=(y_ci_range[:, is_out] if y_ci_range is not None else None),
                    fmt="o",
                    color=outlier_color,
                    alpha=0.5,
                )

        # Pearson r for all and inliers
        rho_all, _ = pearsonr(x, y)
        rho_all_ci = fisher_r_ci(rho_all, n, alpha=ci_alpha)
        rho_in = float("nan")
        rho_in_ci = (float("nan"), float("nan"))
        if inliers.sum() >= 3:
            rho_in, _ = pearsonr(x[inliers], y[inliers])
            rho_in_ci = fisher_r_ci(rho_in, int(inliers.sum()), alpha=ci_alpha)

        # Axes labels and title
        if include_stats_in_title:
            # Copy subtitle formatting verbatim
            n_all = n
            n_in = int((~is_out).sum())
            group_str = f" (Group {group_id})" if len(unique_group_ids) > 1 else ""
            if mark_outliers:
                lines.append(
                    f"All{group_str}: ρ = {rho_all:.3f} [95% CI: {rho_all_ci[0]:.3f}, {rho_all_ci[1]:.3f}] (n = {n_all})"
                )
                if n_in >= 3:
                    lines.append(
                        f"w/o Outliers{group_str}: ρ = {rho_in:.3f} [95% CI: {rho_in_ci[0]:.3f}, {rho_in_ci[1]:.3f}] (n = {n_in})"
                    )
            else:
                lines.append(
                    f"{group_str} ρ = {rho_all:.3f} [95% CI: {rho_all_ci[0]:.3f}, {rho_all_ci[1]:.3f}] (n = {n_all})"
                )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")

    ax.set_title("\n".join(lines))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if len(unique_group_ids) == 1:
        return {
            "coef": coef,
            "intercept": intercept,
            "cooks_threshold": thr,
            "outlier_mask": is_out,
            "rho_all": float(rho_all),
            "rho_all_ci": (float(rho_all_ci[0]), float(rho_all_ci[1])),
            "rho_in": float(rho_in),
            "rho_in_ci": (float(rho_in_ci[0]), float(rho_in_ci[1])),
        }


def logistic_coef_loss_ci(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = False,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    """
    Bootstraps the coefficients of a logistic regression model.

    :param X: The independent variables of the logistic regression model.
    :type X: np.ndarray
    :param y: The response variable of the logistic regression model.
    :type y: np.ndarray
    :param n_bootstrap: The number of bootstrap iterations, defaults to 2000.
    :type n_bootstrap: int, optional
    :param alpha: The significance level for the confidence intervals, defaults to 0.05.
    :type alpha: float, optional
    :return: The coefficients of the logistic regression model, the Wald confidence intervals of the coefficients, the log loss of the model, and the bootstrap confidence intervals for the log loss.
    :rtype: tuple[np.ndarray, np.ndarray, float, tuple[float, float]]
    """
    silent = int(os.environ.get("DEBUG", "0")) < 2

    X = np.array(X)
    y = np.array(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 2:
        assert y.shape[1] == 1 or y.shape[0] == 1, "y must be a 1D array"
        y = y.flatten()

    if not silent:
        print("--- Data ---")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")

    # --- 2. Fit Model with Statsmodels (for detailed output & Wald CIs) ---
    if fit_intercept:
        X_const = sm.add_constant(X, prepend=True)
    else:
        X_const = X
    sm_model = sm.Logit(y, X_const)
    sm_results = sm_model.fit(disp=0)

    # Extract point estimates and Wald CIs
    sm_params = sm_results.params
    sm_wald_ci = sm_results.conf_int(alpha=alpha)
    sm_logloss = -sm_results.llf / sm_results.nobs

    if not silent:
        print("\n--- Statsmodels Fit ---")
        print(sm_results.summary())
        print(f"\nStatsmodels Point Estimate Log Loss: {sm_logloss:.4f}")

    # Function for Loss CIs
    def bootstrap_logistic_loss_ci(X, y, alpha=0.05, initial_params=None):
        n_samples = len(y)
        bootstrap_losses = np.zeros(n_bootstrap)
        if fit_intercept:
            X_const_orig = sm.add_constant(X, prepend=True)
        else:
            X_const_orig = X

        fit_options = {"disp": 0}
        if initial_params is not None:
            fit_options["start_params"] = initial_params

        for i in tqdm(range(n_bootstrap), desc="CI Bootstrap", unit="iter"):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot_const = X_const_orig[indices]
            y_boot = y[indices]
            try:
                model_boot = sm.Logit(y_boot, X_boot_const)
                results_boot = model_boot.fit(**fit_options)
                pred_proba_orig = results_boot.predict(X_const_orig)
                epsilon = 1e-15
                pred_proba_orig = np.clip(pred_proba_orig, epsilon, 1 - epsilon)
                loss = log_loss(y, pred_proba_orig)
                bootstrap_losses[i] = loss
            except Exception:
                bootstrap_losses[i] = np.nan

        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100

        if np.all(np.isnan(bootstrap_losses)):
            print("WARNING: All bootstrap iterations failed for loss calculation.")
            return np.nan, np.nan

        ci_lower = np.nanpercentile(bootstrap_losses, lower_p)
        ci_upper = np.nanpercentile(bootstrap_losses, upper_p)
        return ci_lower, ci_upper

    # --- Execute Bootstrapping ---
    if not silent:
        print(f"\n--- Running {n_bootstrap} Bootstrap Iterations ---")

    bootstrap_loss_ci = bootstrap_logistic_loss_ci(X, y, initial_params=sm_results.params)

    # Coefficients Table
    if not silent:
        print("\n--- Table 1: Comparison of Coefficient Confidence Intervals (95%) ---")
        comparison_data = {
            "Parameters": sm_params.tolist(),
            "Coef CI (lower)": sm_wald_ci[:, 0].tolist(),
            "Coef CI (upper)": sm_wald_ci[:, 1].tolist(),
        }
        print(pd.DataFrame(comparison_data))
        print("------------------------------------------------------------------")

    # Log Loss CI
    if not silent:
        print("\n--- Cross-Entropy Loss (Log Loss) ---")
        print(f"Point Estimate (Statsmodels): {sm_logloss:.4f}")
        if not np.any(np.isnan(bootstrap_loss_ci)):
            print(f"Bootstrap 95% CI: ({bootstrap_loss_ci[0]:.4f}, {bootstrap_loss_ci[1]:.4f})")
        else:
            print("Bootstrap CI for Log Loss could not be computed.")

    return sm_params.tolist(), sm_wald_ci.tolist(), sm_logloss, bootstrap_loss_ci
