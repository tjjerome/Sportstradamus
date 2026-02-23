#!/usr/bin/env python
"""
Compare StudentT vs Mixture of 2 Gaussians vs Single Gaussian

Tests which distribution fits best on skewed datasets:
- NFL passing yards
- NFL rushing yards  
- NBA PTS

This compares:
1. Gaussian (baseline)
2. StudentT (tail-heavy alternative)
3. Mixture of 2 Gaussians (bimodal alternative)
4. SplineFlow (maximum flexibility)
"""

import numpy as np
import pandas as pd
import importlib.resources as pkg_resources
from sportstradamus import data
from scipy.stats import norm, t as scipy_t, skew as scipy_skew
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("DISTRIBUTION COMPARISON TEST: GAUSSIAN vs STUDENTT vs MIXTURE")
print("=" * 80)

# Markets to test
test_markets = {
    "NFL": ["passing-yards", "rushing-yards"],
    "NBA": ["PTS"]
}

results = []

for league, markets in test_markets.items():
    print(f"\n{'='*80}")
    print(f"LEAGUE: {league}")
    print(f"{'='*80}")
    
    for market in markets:
        filename = f"{league}_{market}".replace("-", "-").replace(" ", "-")
        filepath = pkg_resources.files(data) / f"training_data/{filename}.csv"
        
        if not os.path.isfile(filepath):
            print(f"\n⚠ No training data found for {league} {market}")
            continue
        
        print(f"\n{'-'*80}")
        print(f"Market: {league} - {market}")
        print(f"{'-'*80}")
        
        # Load and prepare data
        try:
            M = pd.read_csv(filepath, index_col=0)
            print(f"✓ Loaded {len(M)} records")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            continue
        
        # Get sample
        result_data = M['Result'].values
        print(f"\nData Statistics:")
        print(f"  Mean: {result_data.mean():.3f}")
        print(f"  Std: {result_data.std():.3f}")
        print(f"  Skewness: {scipy_skew(result_data):.3f}")
        
        # Use subset for faster calculation
        np.random.seed(42)
        sample_idx = np.random.choice(len(M), size=min(500, len(M)), replace=False)
        y_val = M.iloc[sample_idx]['Result'].values
        
        dist_results = {}
        
        print(f"\nTesting distributions on {len(y_val)} samples:")
        print(f"  {'-'*76}")
        
        # 1. Single Gaussian
        mu_g, sigma_g = y_val.mean(), y_val.std()
        nll_gaussian = -np.mean(norm.logpdf(y_val, mu_g, sigma_g))
        dist_results["Gaussian"] = {"NLL": nll_gaussian, "n_params": 2}
        print(f"  Gaussian           - NLL: {nll_gaussian:.4f} (2 params)")
        
        # 2. StudentT
        try:
            def neg_loglik_t(params):
                df, loc, scale = params
                if scale <= 0 or df <= 0:
                    return 1e10
                return -np.mean(scipy_t.logpdf(y_val, df=np.clip(df, 1, 100), loc=loc, scale=scale))
            
            x0 = [10, mu_g, sigma_g]
            res = minimize(neg_loglik_t, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            nll_student = res.fun
            dist_results["StudentT"] = {"NLL": nll_student, "n_params": 3}
            print(f"  StudentT           - NLL: {nll_student:.4f} (3 params) [df={res.x[0]:.2f}]")
        except Exception as e:
            print(f"  StudentT           - Error: {str(e)[:40]}")
            dist_results["StudentT"] = {"NLL": float('inf'), "n_params": 3}
        
        # 3. Mixture of 2 Gaussians
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
            gmm.fit(y_val.reshape(-1, 1))
            
            # Calculate log likelihood
            log_likelihoods = gmm.score_samples(y_val.reshape(-1, 1))
            nll_mixture = -np.mean(log_likelihoods)
            
            # Extract fitted parameters for display
            weights = gmm.weights_
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            
            dist_results["Mixture2G"] = {"NLL": nll_mixture, "n_params": 5}
            print(f"  Mixture of 2 Gauss - NLL: {nll_mixture:.4f} (5 params)")
            print(f"    └─ Weight1: {weights[0]:.3f}, μ₁: {means[0]:.2f}, σ₁: {stds[0]:.2f}")
            print(f"    └─ Weight2: {weights[1]:.3f}, μ₂: {means[1]:.2f}, σ₂: {stds[1]:.2f}")
        except Exception as e:
            print(f"  Mixture of 2 Gauss - Error: {str(e)[:40]}")
            dist_results["Mixture2G"] = {"NLL": float('inf'), "n_params": 5}
        
        # 4. Calculate AIC for model comparison
        print(f"\n  Model Comparison (lower is better):")
        print(f"  {'-'*76}")
        aic_scores = {}
        for dist_name, result in dist_results.items():
            n_params = result['n_params']
            nll = result['NLL']
            aic = 2 * n_params + 2 * nll * len(y_val)
            aic_scores[dist_name] = aic
            print(f"  {dist_name:15} - AIC: {aic:10.1f}, NLL: {nll:8.4f}")
        
        # Best by NLL and AIC
        best_nll = min(dist_results.items(), key=lambda x: x[1]["NLL"])
        best_aic = min(aic_scores.items(), key=lambda x: x[1])
        
        print(f"\n  Results:")
        print(f"  ★ Best fit (NLL):  {best_nll[0]}")
        print(f"  ★ Best model (AIC): {best_aic[0]}")
        
        # Improvements
        gaussian_nll = dist_results.get("Gaussian", {}).get("NLL", float('inf'))
        student_nll = dist_results.get("StudentT", {}).get("NLL", float('inf'))
        mixture_nll = dist_results.get("Mixture2G", {}).get("NLL", float('inf'))
        
        if student_nll < float('inf') and gaussian_nll < float('inf'):
            student_improvement = (gaussian_nll - student_nll) / gaussian_nll * 100
            print(f"\n  StudentT vs Gaussian: {student_improvement:+.2f}%")
        
        if mixture_nll < float('inf') and gaussian_nll < float('inf'):
            mixture_improvement = (gaussian_nll - mixture_nll) / gaussian_nll * 100
            print(f"  Mixture vs Gaussian:  {mixture_improvement:+.2f}%")
        
        if student_nll < float('inf') and mixture_nll < float('inf'):
            vs_diff = (student_nll - mixture_nll) / student_nll * 100
            print(f"  Mixture vs StudentT:  {vs_diff:+.2f}%")
        
        results.append({
            "league": league,
            "market": market,
            "best_nll": best_nll[0],
            "best_aic": best_aic[0],
            "skewness": scipy_skew(result_data),
            "nll_scores": {k: v["NLL"] for k, v in dist_results.items()},
            "aic_scores": aic_scores
        })

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATION")
print("=" * 80)

for result in results:
    print(f"\n{result['league']} - {result['market']}:")
    print(f"  Skewness: {result['skewness']:+.3f}")
    print(f"  Best by NLL: {result['best_nll']}")
    print(f"  Best by AIC: {result['best_aic']}")
    print(f"  Models ranked by NLL:")
    for dist, nll in sorted(result['nll_scores'].items(), key=lambda x: x[1]):
        marker = "★" if dist == result['best_nll'] else " "
        print(f"    {marker} {dist:15}: {nll:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
Comparing three approaches for skewed data:

1. GAUSSIAN (baseline)
   - Pros: Simple, 2 parameters
   - Cons: Assumes symmetry, poor for skewed data
   
2. STUDENTT (tail-robust)
   - Pros: 3 parameters, handles both skew and outliers via df
   - Cons: Requires optimization, less interpretable
   
3. MIXTURE OF 2 GAUSSIANS (bimodal)
   - Pros: Can capture two distinct populations/modes
   - Cons: 5 parameters (may overfit), slower, less stable
   - Use case: Data with clear bimodal/bimodal structure
   
AIC METRIC: Balances fit quality with model complexity
- Lower AIC = Better model (accounts for overfitting)
- Penalizes extra parameters: AIC = 2k + 2n*NLL
""")

print("=" * 80)
print("Test completed!")
print("=" * 80)
