"""
Tests for CUPED (Controlled-experiment Using Pre-Experiment Data) implementation.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
import warnings

from src.causal_methods.cuped import CUPED, load_and_analyze_cuped


@pytest.fixture
def sample_data():
    """Generate sample experimental data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Pre-experiment covariates
    baseline_outcome = np.random.normal(10, 3, n)
    user_engagement = np.random.normal(50, 15, n)
    user_age = np.random.normal(35, 10, n)
    
    # Random treatment assignment (ensures balance)
    treatment = np.random.binomial(1, 0.5, n)
    
    # Post-experiment outcome (correlated with baseline)
    # True treatment effect = 2.0
    outcome = (0.7 * baseline_outcome + 
              0.1 * user_engagement + 
              0.05 * user_age +
              2.0 * treatment +  # True treatment effect
              np.random.normal(0, 2, n))
    
    data = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'baseline_outcome': baseline_outcome,
        'user_engagement': user_engagement,
        'user_age': user_age
    })
    
    return data


@pytest.fixture
def cuped_analyzer(sample_data):
    """Create CUPED analyzer with sample data."""
    return CUPED(sample_data, random_state=42)


class TestCUPEDInitialization:
    """Test CUPED initialization and basic functionality."""
    
    def test_initialization(self, sample_data):
        """Test CUPED initialization."""
        cuped = CUPED(sample_data, random_state=42)
        
        assert cuped.data.shape == sample_data.shape
        assert cuped.random_state == 42
        assert len(cuped.results) == 0
        assert len(cuped.cuped_adjustments) == 0
    
    def test_initialization_without_random_state(self, sample_data):
        """Test initialization without random state."""
        cuped = CUPED(sample_data)
        
        assert cuped.random_state is None
        assert cuped.data.shape == sample_data.shape


class TestCUPEDAdjustmentEstimation:
    """Test CUPED adjustment coefficient estimation."""
    
    def test_estimate_cuped_adjustment_basic(self, cuped_analyzer):
        """Test basic CUPED adjustment estimation."""
        adjustment_info = cuped_analyzer.estimate_cuped_adjustment(
            outcome_col='outcome',
            covariate_cols=['baseline_outcome', 'user_engagement']
        )
        
        assert 'theta' in adjustment_info
        assert 'covariate_means' in adjustment_info
        assert 'r2' in adjustment_info
        assert 'variance_reduction' in adjustment_info
        assert len(adjustment_info['theta']) == 2
        assert adjustment_info['r2'] > 0  # Should have some predictive power
    
    def test_estimate_cuped_adjustment_with_treatment(self, cuped_analyzer):
        """Test CUPED adjustment estimation with treatment column."""
        adjustment_info = cuped_analyzer.estimate_cuped_adjustment(
            outcome_col='outcome',
            covariate_cols=['baseline_outcome'],
            treatment_col='treatment'
        )
        
        assert 'balance_check' in adjustment_info
        assert 'baseline_outcome' in adjustment_info['balance_check']
        
        # Check balance metrics
        balance = adjustment_info['balance_check']['baseline_outcome']
        assert 'std_diff' in balance
        assert 'p_value' in balance
        assert 'balanced' in balance
    
    def test_estimate_cuped_adjustment_different_methods(self, cuped_analyzer):
        """Test different adjustment methods."""
        methods = ['ols', 'ridge', 'lasso']
        
        for method in methods:
            adjustment_info = cuped_analyzer.estimate_cuped_adjustment(
                outcome_col='outcome',
                covariate_cols=['baseline_outcome'],
                method=method
            )
            
            assert adjustment_info['method'] == method
            assert 'theta' in adjustment_info
            assert len(adjustment_info['theta']) == 1
    
    def test_estimate_cuped_adjustment_invalid_method(self, cuped_analyzer):
        """Test invalid adjustment method."""
        with pytest.raises(ValueError, match="Unknown method"):
            cuped_analyzer.estimate_cuped_adjustment(
                outcome_col='outcome',
                covariate_cols=['baseline_outcome'],
                method='invalid_method'
            )
    
    def test_estimate_cuped_adjustment_missing_values(self, sample_data):
        """Test handling of missing values in covariates."""
        # Introduce missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:10, 'baseline_outcome'] = np.nan
        
        cuped = CUPED(data_with_missing)
        
        with warnings.catch_warnings(record=True) as w:
            adjustment_info = cuped.estimate_cuped_adjustment(
                outcome_col='outcome',
                covariate_cols=['baseline_outcome']
            )
            assert len(w) == 1
            assert "Missing values in covariates" in str(w[0].message)
        
        assert 'theta' in adjustment_info


class TestCUPEDAdjustmentApplication:
    """Test application of CUPED adjustments."""
    
    def test_apply_cuped_adjustment(self, cuped_analyzer):
        """Test applying CUPED adjustment."""
        # First estimate adjustment
        adjustment_info = cuped_analyzer.estimate_cuped_adjustment(
            outcome_col='outcome',
            covariate_cols=['baseline_outcome']
        )
        
        # Apply adjustment
        cuped_outcome = cuped_analyzer.apply_cuped_adjustment('outcome')
        
        assert len(cuped_outcome) == len(cuped_analyzer.data)
        assert cuped_outcome.notna().sum() > 0
        
        # CUPED-adjusted outcome should have lower variance
        original_var = cuped_analyzer.data['outcome'].var()
        cuped_var = cuped_outcome.var()
        assert cuped_var < original_var
    
    def test_apply_cuped_adjustment_without_estimation(self, cuped_analyzer):
        """Test applying adjustment without prior estimation."""
        with pytest.raises(ValueError, match="No adjustment computed"):
            cuped_analyzer.apply_cuped_adjustment('outcome')
    
    def test_apply_cuped_adjustment_with_provided_info(self, cuped_analyzer):
        """Test applying adjustment with provided adjustment info."""
        # Create mock adjustment info
        adjustment_info = {
            'theta': np.array([0.5]),
            'covariate_means': pd.Series([10.0], index=['baseline_outcome']),
            'covariate_cols': ['baseline_outcome']
        }
        
        cuped_outcome = cuped_analyzer.apply_cuped_adjustment(
            'outcome', adjustment_info
        )
        
        assert len(cuped_outcome) == len(cuped_analyzer.data)


class TestCUPEDTreatmentEffects:
    """Test treatment effect estimation with CUPED."""
    
    def test_estimate_treatment_effects_basic(self, cuped_analyzer):
        """Test basic treatment effect estimation."""
        results = cuped_analyzer.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline_outcome']
        )
        
        assert 'original' in results
        assert 'cuped' in results
        assert 'summary' in results
        
        # Check original results
        orig = results['original']
        assert 'ate' in orig
        assert 'se' in orig
        assert 'p_value' in orig
        
        # Check CUPED results
        cuped = results['cuped']
        assert 'ate' in cuped
        assert 'se' in cuped
        assert 'p_value' in cuped
        
        # CUPED should have lower standard error
        assert cuped['se'] < orig['se']
        
        # Check summary
        summary = results['summary']
        assert 'variance_reduction' in summary
        assert 'power_improvement' in summary
        assert summary['variance_reduction'] > 0
        assert summary['power_improvement'] > 1.0
    
    def test_estimate_treatment_effects_multiple_covariates(self, cuped_analyzer):
        """Test treatment effects with multiple covariates."""
        results = cuped_analyzer.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline_outcome', 'user_engagement', 'user_age']
        )
        
        assert 'summary' in results
        assert len(results['summary']['covariate_cols']) == 3
        
        # Should have even better variance reduction with more relevant covariates
        assert results['summary']['variance_reduction'] > 0
    
    def test_estimate_treatment_effects_different_confidence_levels(self, cuped_analyzer):
        """Test different confidence levels."""
        confidence_levels = [0.90, 0.95, 0.99]
        
        for conf_level in confidence_levels:
            results = cuped_analyzer.estimate_treatment_effects(
                outcome_col='outcome',
                treatment_col='treatment',
                covariate_cols=['baseline_outcome'],
                confidence_level=conf_level
            )
            
            # Higher confidence level should give wider intervals
            orig_width = (results['original']['ci_upper'] - 
                         results['original']['ci_lower'])
            cuped_width = (results['cuped']['ci_upper'] - 
                          results['cuped']['ci_lower'])
            
            assert orig_width > 0
            assert cuped_width > 0
            assert cuped_width < orig_width  # CUPED should have narrower CI


class TestCUPEDVisualization:
    """Test CUPED visualization functionality."""
    
    def test_plot_cuped_comparison(self, cuped_analyzer):
        """Test CUPED comparison plot."""
        # First estimate treatment effects
        cuped_analyzer.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline_outcome']
        )
        
        # Test plotting
        fig = cuped_analyzer.plot_cuped_comparison(
            outcome_col='outcome',
            treatment_col='treatment'
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # Should have 6 subplots
        plt.close(fig)
    
    def test_plot_cuped_comparison_without_results(self, cuped_analyzer):
        """Test plotting without prior results."""
        with pytest.raises(ValueError, match="No results found"):
            cuped_analyzer.plot_cuped_comparison(
                outcome_col='outcome',
                treatment_col='treatment'
            )


class TestCUPEDReporting:
    """Test CUPED summary report generation."""
    
    def test_generate_summary_report(self, cuped_analyzer):
        """Test summary report generation."""
        # First estimate treatment effects
        cuped_analyzer.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline_outcome']
        )
        
        report = cuped_analyzer.generate_summary_report('outcome')
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "CUPED" in report
        assert "ADJUSTMENT DETAILS" in report
        assert "TREATMENT EFFECT RESULTS" in report
        assert "IMPROVEMENTS" in report
    
    def test_generate_summary_report_without_results(self, cuped_analyzer):
        """Test report generation without results."""
        with pytest.raises(ValueError, match="No results found"):
            cuped_analyzer.generate_summary_report('outcome')


class TestCUPEDBalanceChecks:
    """Test covariate balance checking functionality."""
    
    def test_check_covariate_balance_balanced_data(self, cuped_analyzer):
        """Test balance check with balanced data."""
        # Create perfectly balanced data
        data_balanced = cuped_analyzer.data.copy()
        n_half = len(data_balanced) // 2
        data_balanced['treatment'] = [0] * n_half + [1] * (len(data_balanced) - n_half)
        
        cuped_balanced = CUPED(data_balanced)
        covariates = data_balanced[['baseline_outcome']]
        treatment = data_balanced['treatment']
        
        balance_results = cuped_balanced._check_covariate_balance(covariates, treatment)
        
        assert 'baseline_outcome' in balance_results
        balance = balance_results['baseline_outcome']
        assert 'std_diff' in balance
        assert 'balanced' in balance
    
    def test_check_covariate_balance_imbalanced_data(self):
        """Test balance check with imbalanced data."""
        np.random.seed(42)
        n = 100
        
        # Create imbalanced data
        treatment = np.concatenate([np.zeros(20), np.ones(80)])
        covariate = np.concatenate([
            np.random.normal(5, 1, 20),    # Control group (lower values)
            np.random.normal(15, 1, 80)   # Treatment group (higher values)
        ])
        
        data_imbalanced = pd.DataFrame({
            'treatment': treatment,
            'covariate': covariate
        })
        
        cuped_imbalanced = CUPED(data_imbalanced)
        covariates = data_imbalanced[['covariate']]
        
        balance_results = cuped_imbalanced._check_covariate_balance(
            covariates, data_imbalanced['treatment']
        )
        
        balance = balance_results['covariate']
        assert abs(balance['std_diff']) > 0.1  # Should be imbalanced
        assert not balance['balanced']


class TestCUPEDEdgeCases:
    """Test CUPED edge cases and error handling."""
    
    def test_small_sample_size(self):
        """Test CUPED with very small sample size."""
        np.random.seed(42)
        small_data = pd.DataFrame({
            'outcome': np.random.normal(0, 1, 20),
            'treatment': [0] * 10 + [1] * 10,
            'covariate': np.random.normal(0, 1, 20)
        })
        
        cuped = CUPED(small_data)
        
        # Should still work but with warnings about small sample
        results = cuped.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['covariate']
        )
        
        assert 'original' in results
        assert 'cuped' in results
    
    def test_no_treatment_variation(self):
        """Test CUPED when all units have same treatment."""
        data_no_variation = pd.DataFrame({
            'outcome': np.random.normal(0, 1, 100),
            'treatment': [1] * 100,  # All treated
            'covariate': np.random.normal(0, 1, 100)
        })
        
        cuped = CUPED(data_no_variation)
        
        # Should handle gracefully (though not meaningful)
        with pytest.raises(Exception):  # Expect some kind of error
            cuped.estimate_treatment_effects(
                outcome_col='outcome',
                treatment_col='treatment',
                covariate_cols=['covariate']
            )
    
    def test_perfect_correlation_covariate(self):
        """Test CUPED with perfectly correlated covariate."""
        np.random.seed(42)
        treatment = np.random.binomial(1, 0.5, 100)
        outcome = np.random.normal(0, 1, 100)
        
        # Perfect correlation with outcome
        covariate = outcome + np.random.normal(0, 0.001, 100)
        
        data_perfect_corr = pd.DataFrame({
            'outcome': outcome,
            'treatment': treatment,
            'covariate': covariate
        })
        
        cuped = CUPED(data_perfect_corr)
        
        results = cuped.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['covariate']
        )
        
        # Should achieve very high variance reduction
        assert results['summary']['variance_reduction'] > 0.9


class TestCUPEDUtilityFunctions:
    """Test utility functions and integration."""
    
    def test_load_and_analyze_cuped_csv(self, sample_data, tmp_path):
        """Test loading and analyzing data from CSV."""
        # Save sample data to CSV
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Test loading and analysis
        cuped = load_and_analyze_cuped(
            data_path=str(csv_path),
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline_outcome']
        )
        
        assert isinstance(cuped, CUPED)
        assert 'outcome' in cuped.results
    
    def test_load_and_analyze_cuped_unsupported_format(self, tmp_path):
        """Test loading unsupported file format."""
        # Create a .txt file
        txt_path = tmp_path / "test_data.txt"
        txt_path.write_text("dummy content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_and_analyze_cuped(
                data_path=str(txt_path),
                outcome_col='outcome',
                treatment_col='treatment',
                covariate_cols=['baseline_outcome']
            )


class TestCUPEDIntegration:
    """Integration tests for CUPED functionality."""
    
    def test_full_cuped_workflow(self, cuped_analyzer):
        """Test complete CUPED workflow."""
        # Step 1: Estimate adjustment
        adjustment_info = cuped_analyzer.estimate_cuped_adjustment(
            outcome_col='outcome',
            covariate_cols=['baseline_outcome', 'user_engagement'],
            treatment_col='treatment'
        )
        
        assert adjustment_info['r2'] > 0
        
        # Step 2: Apply adjustment
        cuped_outcome = cuped_analyzer.apply_cuped_adjustment('outcome')
        assert cuped_outcome.notna().sum() > 0
        
        # Step 3: Estimate treatment effects
        results = cuped_analyzer.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline_outcome', 'user_engagement']
        )
        
        assert results['summary']['variance_reduction'] > 0
        
        # Step 4: Generate report
        report = cuped_analyzer.generate_summary_report('outcome')
        assert len(report) > 0
        
        # Step 5: Create visualization
        fig = cuped_analyzer.plot_cuped_comparison('outcome', 'treatment')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_cuped_with_weak_covariates(self):
        """Test CUPED performance with weak covariates."""
        np.random.seed(42)
        n = 500
        
        # Create data with weak covariate correlation
        treatment = np.random.binomial(1, 0.5, n)
        weak_covariate = np.random.normal(0, 1, n)  # Uncorrelated
        outcome = 2.0 * treatment + np.random.normal(0, 1, n)
        
        data_weak = pd.DataFrame({
            'outcome': outcome,
            'treatment': treatment,
            'weak_covariate': weak_covariate
        })
        
        cuped = CUPED(data_weak)
        results = cuped.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['weak_covariate']
        )
        
        # Should have minimal variance reduction
        assert results['summary']['variance_reduction'] < 0.1
        
        # Generate report should include warnings
        report = cuped.generate_summary_report('outcome')
        assert "Limited variance reduction" in report or "Low R²" in report
    
    def test_cuped_robustness_to_outliers(self):
        """Test CUPED robustness to outliers."""
        np.random.seed(42)
        n = 200
        
        treatment = np.random.binomial(1, 0.5, n)
        baseline = np.random.normal(10, 2, n)
        outcome = 0.8 * baseline + 1.5 * treatment + np.random.normal(0, 1, n)
        
        # Add outliers
        outcome[0] = 100  # Extreme outlier
        baseline[1] = -50  # Extreme outlier
        
        data_outliers = pd.DataFrame({
            'outcome': outcome,
            'treatment': treatment,
            'baseline': baseline
        })
        
        cuped = CUPED(data_outliers)
        results = cuped.estimate_treatment_effects(
            outcome_col='outcome',
            treatment_col='treatment',
            covariate_cols=['baseline']
        )
        
        # Should still provide some variance reduction
        assert results['summary']['variance_reduction'] > 0 