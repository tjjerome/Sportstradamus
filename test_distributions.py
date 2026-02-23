#!/usr/bin/env python
"""
Test script to validate the new distribution support in train.py
Run with: poetry run python test_distributions.py
"""

import numpy as np
import pandas as pd
from scipy import stats

# Test imports
print("=" * 60)
print("Testing Distribution Support")
print("=" * 60)

try:
    from lightgbmlss.distributions import (
        Gaussian, Poisson, StudentT, SplineFlow
    )
    print("✓ All distributions imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

try:
    from sportstradamus.train import distributions, _set_model_start_values
    print("✓ Train module helper functions imported successfully")
except ImportError as e:
    print(f"✗ Train module import error: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Available Distributions")
print("=" * 60)

for dist_name, dist_obj in distributions.items():
    n_params = dist_obj.n_dist_param
    print(f"  {dist_name:15} - {n_params} parameter(s)")

# Test with sample data
print("\n" + "=" * 60)
print("Testing with Sample Data")
print("=" * 60)

# Create sample training data
np.random.seed(42)
n_samples = 1000

# Simulate skewed Gaussian data (your use case)
# Using skew_normal approximation
skew = 1.5
data = stats.skewnorm.rvs(skew, size=n_samples, loc=10, scale=3)
data = np.clip(data, 0, 100)  # Clip to reasonable sports values

X = pd.DataFrame({
    'MeanYr': data.mean() * np.ones(n_samples),
    'STDYr': data.std() * np.ones(n_samples),
    'Feature1': np.random.randn(n_samples),
    'Feature2': np.random.randn(n_samples),
})

print(f"\nSample data characteristics:")
print(f"  N = {len(data)}")
print(f"  Mean = {data.mean():.2f}")
print(f"  Std = {data.std():.2f}")
print(f"  Skewness = {stats.skew(data):.2f}")
print(f"  Kurtosis = {stats.kurtosis(data):.2f}")

# Test each distribution's parameter count
print("\n" + "=" * 60)
print("Distribution Parameter Verification")
print("=" * 60)

from lightgbmlss.model import LightGBMLSS

dist_info = {
    "Poisson": {"n_params": 1, "description": "Rate"},
    "Gaussian": {"n_params": 2, "description": "Mean, Std"},
    "StudentT": {"n_params": 3, "description": "DF, Mean, Std"},
    "SplineFlow": {"n_params": 31, "description": "Spline coefficients"},
}

for dist_name, info in dist_info.items():
    dist_obj = distributions[dist_name]
    actual_params = dist_obj.n_dist_param
    expected_params = info["n_params"]
    status = "✓" if actual_params == expected_params else "✗"
    print(f"{status} {dist_name:15} - Expected: {expected_params:2}, Actual: {actual_params:2}")
    print(f"   └─ {info['description']}")

# Test start value initialization
print("\n" + "=" * 60)
print("Testing Start Value Initialization")
print("=" * 60)

from lightgbmlss.model import LightGBMLSS
import lightgbm as lgb

class MockModel:
    """Mock model for testing start_values"""
    def __init__(self, n_dist_param):
        self.n_dist_param = n_dist_param
        self.start_values = None

# Create a small dataset for testing
X_sample = X.head(100)
y_sample = data[:100]

dtrain = lgb.Dataset(X_sample[['MeanYr', 'STDYr']], label=y_sample)

for dist_name in ["Poisson", "Gaussian", "StudentT", "SplineFlow"]:
    try:
        model = LightGBMLSS(distributions[dist_name])
        _set_model_start_values(model, dist_name, X_sample)
        sv = model.start_values
        if sv is not None:
            print(f"✓ {dist_name:15} - Start values shape: {sv.shape}")
        else:
            print(f"? {dist_name:15} - Start values not set (may be normal)")
    except Exception as e:
        print(f"✗ {dist_name:15} - Error: {e}")

# Test with skewed data statistics
print("\n" + "=" * 60)
print("Why StudentT for Skewed Data?")
print("=" * 60)

print(f"""
Your Gaussian data characteristics:
  • Mean: {data.mean():.2f}
  • Std: {data.std():.2f}
  • Skewness: {stats.skew(data):.2f} ({'right-skewed' if stats.skew(data) > 0 else 'left-skewed'})
  • Kurtosis: {stats.kurtosis(data):.2f} ({'heavy-tailed' if stats.kurtosis(data) > 0 else 'light-tailed'})

StudentT Advantage:
  • Degrees of freedom parameter (df) handles tail behavior
  • df << 30 → Heavy tails (robust to outliers)
  • df ≈ 30 → Similar to Gaussian
  • Naturally accommodates skewness through heavy tails

Recommendation:
  Use StudentT for this data distribution type!
""")

print("\n" + "=" * 60)
print("All Tests Completed!")
print("=" * 60)
