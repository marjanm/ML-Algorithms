"""
Data Generators
================
Centralized synthetic data generation for all model demos.

Usage:
    from data_generators.classification_data import generate_synthetic_data
    from data_generators.regression_data import generate_regression_data
    from data_generators.clustering_data import generate_clustering_data
    from data_generators.timeseries_data import generate_timeseries_data
    from data_generators.anomaly_data import generate_anomaly_data

Backward compat (old import still works):
    from data_generators import generate_synthetic_data
"""

from .classification_data import generate_synthetic_data
from .regression_data import generate_regression_data
from .clustering_data import generate_clustering_data
from .timeseries_data import generate_timeseries_data
from .anomaly_data import generate_anomaly_data
