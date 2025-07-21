#!/usr/bin/env python3
"""Debug script to investigate NaN p-value issue in PSM."""

import sys
sys.path.append('src')

from causal_methods.psm import PropensityScoreMatching
from data_simulation import generate_and_save_data
import pandas as pd
import numpy as np

def debug_psm_pvalues():
    """Debug PSM p-value calculation."""
    print("=" * 50)
    print("DEBUGGING PSM P-VALUE ISSUE")
    print("=" * 50)
    
    # Generate test data
    print("1. Generating test data...")
    df = generate_and_save_data('test_data.csv', n_users=200, config_path='config/simulation_config.yaml')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Treatment rate: {df['used_smart_assistant'].mean():.2%}")
    
    # Run PSM analysis
    print("\n2. Running PSM analysis...")
    psm = PropensityScoreMatching(df)
    
    # Estimate propensity scores
    ps_results = psm.estimate_propensity_scores()
    print(f"   Propensity scores estimated: {len(psm.propensity_scores)}")
    
    # Perform matching
    matching_results = psm.perform_matching(method='nearest_neighbor', caliper=0.15)
    print(f"   Matching rate: {matching_results['matching_rate']:.2%}")
    print(f"   Matched data size: {len(psm.matched_data) if psm.matched_data is not None else 0}")
    
    if psm.matched_data is not None and len(psm.matched_data) > 0:
        print("\n3. Analyzing matched sample...")
        matched_data = psm.matched_data
        treated = matched_data[matched_data['used_smart_assistant'] == 1]
        control = matched_data[matched_data['used_smart_assistant'] == 0]
        
        print(f"   Treated units: {len(treated)}")
        print(f"   Control units: {len(control)}")
        
        # Check the outcome variable
        outcome = 'filed_2024'
        if outcome in matched_data.columns:
            treated_outcome = treated[outcome].dropna()
            control_outcome = control[outcome].dropna()
            
            print(f"\n4. Analyzing outcome variable '{outcome}':")
            print(f"   Treated - N: {len(treated_outcome)}, Mean: {treated_outcome.mean():.4f}, Std: {treated_outcome.std():.4f}")
            print(f"   Control - N: {len(control_outcome)}, Mean: {control_outcome.mean():.4f}, Std: {control_outcome.std():.4f}")
            print(f"   Unique values in treated: {sorted(treated_outcome.unique())}")
            print(f"   Unique values in control: {sorted(control_outcome.unique())}")
            
            # Manual t-test calculation
            print(f"\n5. Manual t-test calculation:")
            if len(treated_outcome) > 1 and len(control_outcome) > 1:
                from scipy import stats
                try:
                    t_stat, p_value = stats.ttest_ind(treated_outcome, control_outcome)
                    print(f"   T-statistic: {t_stat:.4f}")
                    print(f"   P-value: {p_value:.6f}")
                    
                    # Check for equal variances
                    _, levene_p = stats.levene(treated_outcome, control_outcome)
                    print(f"   Levene test p-value (equal variances): {levene_p:.6f}")
                    
                    # Try Welch's t-test
                    t_stat_welch, p_value_welch = stats.ttest_ind(treated_outcome, control_outcome, equal_var=False)
                    print(f"   Welch's t-test p-value: {p_value_welch:.6f}")
                    
                except Exception as e:
                    print(f"   Error in t-test: {e}")
            else:
                print(f"   Insufficient data for t-test")
        
        # Test PSM treatment effect estimation
        print(f"\n6. PSM treatment effect estimation:")
        try:
            effects = psm.estimate_treatment_effects(outcome_cols=outcome)
            effect = effects[outcome]
            print(f"   ATE: {effect['ate']:.4f}")
            print(f"   P-value: {effect['p_value']}")
            print(f"   CI: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
            print(f"   Standard error: {effect['standard_error']:.4f}")
            
            # Check if p-value is NaN and why
            if np.isnan(effect['p_value']):
                print(f"   ❌ P-value is NaN!")
                print(f"   Investigating cause...")
                
                # Re-run the calculation manually
                matched_treated = psm.matched_data[psm.matched_data['used_smart_assistant'] == 1][outcome].dropna()
                matched_control = psm.matched_data[psm.matched_data['used_smart_assistant'] == 0][outcome].dropna()
                
                print(f"   Matched treated values: {matched_treated.values}")
                print(f"   Matched control values: {matched_control.values}")
                
            else:
                print(f"   ✅ P-value is valid: {effect['p_value']:.6f}")
                
        except Exception as e:
            print(f"   Error in treatment effect estimation: {e}")
    else:
        print("   ❌ No matched data available")

if __name__ == "__main__":
    debug_psm_pvalues() 