"""
Classical Time Series Analysis — Demo
=======================================
The stats approach to time series (complements the LSTM demo).

Demonstrates:
  1. Time series decomposition (trend + seasonality + residual)
  2. Stationarity testing (Augmented Dickey-Fuller)
  3. ACF/PACF plots for ARIMA order selection
  4. ARIMA model fitting and forecasting
  5. Comparison: ARIMA vs naive baseline

Run:
    python timeseries_demo.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_generators.timeseries_data import generate_timeseries_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_timeseries_demo():
    lines = [
        "=" * 65,
        "  CLASSICAL TIME SERIES ANALYSIS  —  Demo",
        "=" * 65, "",
    ]

    df = generate_timeseries_data(n_points=500, seasonal_period=50, save_csv=False)
    series = df["value"].values

    lines.append(f"  Data: {len(series)} time steps")
    lines.append(f"  True components: trend (slope=0.05) + seasonality (period=50) + noise")

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller, acf, pacf
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        statsmodels_ok = True
    except ImportError:
        statsmodels_ok = False
        lines += [
            "", "  ⚠ statsmodels not installed. Install with: pip3 install statsmodels",
            "  Skipping ARIMA analysis. Showing decomposition with numpy instead.",
        ]

    if not statsmodels_ok:
        lines += ["", "=" * 65]
        output_text = "\n".join(lines)
        print("\n" + output_text)
        with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
            f.write(output_text)
        return

    # ── 1. Decomposition ──
    lines += ["", "  ── 1. Time Series Decomposition ──"]
    decomp = seasonal_decompose(series, model="additive", period=50)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(series, color="steelblue")
    axes[0].set_title("Original")
    axes[1].plot(decomp.trend, color="coral")
    axes[1].set_title("Trend")
    axes[2].plot(decomp.seasonal, color="green")
    axes[2].set_title("Seasonal (period=50)")
    axes[3].plot(decomp.resid, color="gray")
    axes[3].set_title("Residual")
    plt.suptitle("Additive Decomposition", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    lines.append(f"  [saved] → plots/decomposition.png")

    # ── 2. Stationarity test ──
    lines += ["", "  ── 2. Stationarity Test (Augmented Dickey-Fuller) ──"]

    adf_raw = adfuller(series)
    lines += [
        f"    Raw series:",
        f"      ADF statistic: {adf_raw[0]:.4f}",
        f"      P-value:       {adf_raw[1]:.6f}",
        f"      Stationary?    {'YES' if adf_raw[1] < 0.05 else 'NO — needs differencing'}",
    ]

    series_diff = np.diff(series)
    adf_diff = adfuller(series_diff)
    lines += [
        f"    After 1st differencing:",
        f"      ADF statistic: {adf_diff[0]:.4f}",
        f"      P-value:       {adf_diff[1]:.6f}",
        f"      Stationary?    {'YES' if adf_diff[1] < 0.05 else 'NO'}",
    ]

    # ── 3. ACF/PACF ──
    lines += ["", "  ── 3. ACF / PACF (for ARIMA order selection) ──"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series_diff, lags=60, ax=axes[0])
    axes[0].set_title("ACF (differenced) — helps choose q")
    plot_pacf(series_diff, lags=60, ax=axes[1])
    axes[1].set_title("PACF (differenced) — helps choose p")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "acf_pacf.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/acf_pacf.png")
    lines += [
        "    Rule of thumb:",
        "      p = number of significant PACF lags (AR order)",
        "      d = number of differences needed for stationarity",
        "      q = number of significant ACF lags (MA order)",
    ]

    # ── 4. ARIMA fitting ──
    lines += ["", "  ── 4. ARIMA Model ──"]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    orders_to_try = [(1,1,1), (2,1,1), (2,1,2), (1,1,2)]
    best_aic, best_order, best_model = np.inf, None, None

    for order in orders_to_try:
        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            lines.append(f"    ARIMA{order}: AIC={fitted.aic:.1f}")
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
                best_model = fitted
        except Exception:
            lines.append(f"    ARIMA{order}: failed to converge")

    lines.append(f"    Best: ARIMA{best_order} (AIC={best_aic:.1f})")

    # ── 5. Forecasting ──
    lines += ["", "  ── 5. Forecast vs Actual ──"]
    forecast = best_model.forecast(steps=len(test))

    # naive baseline: predict last training value
    naive = np.full(len(test), train[-1])

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    arima_mae = mean_absolute_error(test, forecast)
    arima_rmse = np.sqrt(mean_squared_error(test, forecast))
    naive_mae = mean_absolute_error(test, naive)
    naive_rmse = np.sqrt(mean_squared_error(test, naive))

    lines += [
        f"    {'Model':15s} | {'MAE':>8s} | {'RMSE':>8s}",
        f"    {'-'*15}-+-{'-'*8}-+-{'-'*8}",
        f"    {'ARIMA':15s} | {arima_mae:8.4f} | {arima_rmse:8.4f}",
        f"    {'Naive (last)':15s} | {naive_mae:8.4f} | {naive_rmse:8.4f}",
        f"    ARIMA improvement: {(1 - arima_mae/naive_mae):.1%} lower MAE",
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(train_size), train, color="steelblue", label="Train")
    test_idx = range(train_size, len(series))
    ax.plot(test_idx, test, color="steelblue", alpha=0.4, label="Test (actual)")
    ax.plot(test_idx, forecast, color="coral", linewidth=2, label=f"ARIMA{best_order} forecast")
    ax.plot(test_idx, naive, color="gray", linestyle="--", label="Naive baseline")
    ax.axvline(x=train_size, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"ARIMA{best_order} Forecast vs Actual (MAE={arima_mae:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "forecast.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/forecast.png")

    lines += [
        "", "  ── Key Concepts ──",
        "    • Stationarity: constant mean & variance over time (required for ARIMA)",
        "    • Differencing (d): subtract previous value to remove trend",
        "    • AR(p): current value depends on p past values (autoregressive)",
        "    • MA(q): current value depends on q past forecast errors (moving average)",
        "    • AIC: lower = better model (balances fit vs complexity)",
        "    • For real data: also consider SARIMA (seasonal ARIMA) or Prophet",
        "", "=" * 65,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_timeseries_demo()
