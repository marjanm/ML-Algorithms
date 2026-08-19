"""
Time Series Data Generator
============================
Generates synthetic time series with trend, seasonality, and noise.
Useful for ARIMA, Prophet, and LSTM demos.
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_timeseries_data(
    n_points: int = 500,
    trend_slope: float = 0.05,       # linear trend per time step
    seasonal_period: int = 50,       # period of the seasonal component
    seasonal_amplitude: float = 1.0, # strength of seasonality
    noise_std: float = 0.3,          # gaussian noise level
    random_state: int = 42,
    save_csv: bool = True,
):
    """Create a synthetic time series with trend + seasonality + noise.

    Returns
    -------
    df : DataFrame with columns ['time', 'value', 'trend', 'seasonal', 'noise']
    """
    np.random.seed(random_state)
    t = np.arange(n_points, dtype=float)

    trend = trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(0, noise_std, n_points)
    value = trend + seasonal + noise

    df = pd.DataFrame({
        "time": t,
        "value": value,
        "trend": trend,
        "seasonal": seasonal,
        "noise": noise,
    })

    if save_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "timeseries_dataset.csv")
        df.to_csv(path, index=False)
        print(f"[timeseries] Saved {len(df)} rows  ->  {path}")

    return df


if __name__ == "__main__":
    df = generate_timeseries_data()
    print(f"Shape: {df.shape}")
    print(df.head())
