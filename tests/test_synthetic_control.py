"""
Test suite for synthetic control method implementation.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.causal_methods.synthetic_control import SyntheticControl, load_and_analyze_synthetic_control


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    np.random.seed(42)
    n = 100
    
    # Create predictors
    data = {
        'user_id': [f'user_{i:03d}' for i in range(n)],
        'filed_2023': np.random.binomial(1, 0.7, n),
        'time_to_complete_2023': np.random.normal(60, 20, n),
        'sessions_2023': np.random.poisson(5, n),
        'age': np.random.randint(18, 65, n),
        'tech_savviness': np.random.uniform(0, 10, n),
        'used_smart_assistant': np.random.binomial(1, 0.4, n)
    }
    
    # Create outcome with some treatment effect
    treatment_effect = 0.1
    base_prob = 0.6 + 0.3 * data['filed_2023'] + 0.01 * data['tech_savviness']
    treatment_boost = treatment_effect * data['used_smart_assistant']
    data['filed_2024'] = np.random.binomial(1, np.clip(base_prob + treatment_boost, 0, 1), n)
    
    return pd.DataFrame(data)


@pytest.fixture
def edge_case_dataset():
    """Create dataset with edge cases for testing robustness."""
    np.random.seed(123)
    n = 50
    
    data = {
        'user_id': [f'user_{i:03d}' for i in range(n)],
        'filed_2023': np.random.binomial(1, 0.5, n),
        'predictor_1': np.random.normal(0, 1, n),
        'predictor_2': np.random.normal(0, 1, n),
        'used_smart_assistant': [1] * 10 + [0] * 40,  # Small treatment group
        'filed_2024': np.random.binomial(1, 0.5, n)
    }
    
    # Add some missing values
    data['predictor_1'][5] = np.nan
    data['predictor_2'][15] = np.nan
    
    return pd.DataFrame(data)


class TestSyntheticControlInitialization:
    """Test synthetic control initialization and basic functionality."""
    
    def test_initialization(self, simple_dataset):
        """Test basic initialization."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        assert sc.data is not None
        assert len(sc.data) == len(simple_dataset)
        assert sc.random_state == 42
        assert sc.synthetic_weights == {}
        assert sc.results == {}
    
    def test_initialization_with_data_copy(self, simple_dataset):
        """Test that data is properly copied."""
        original_data = simple_dataset.copy()
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Modify original data
        simple_dataset.loc[0, 'age'] = 999
        
        # Synthetic control should have original values
        assert sc.data.loc[0, 'age'] != 999
        assert sc.data.loc[0, 'age'] == original_data.loc[0, 'age']
    
    def test_initialization_sets_random_seed(self, simple_dataset):
        """Test that random seed is properly set."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Generate some random numbers
        random_vals_1 = [np.random.random() for _ in range(5)]
        
        # Re-initialize with same seed
        sc2 = SyntheticControl(simple_dataset, random_state=42)
        random_vals_2 = [np.random.random() for _ in range(5)]
        
        # Values should be the same
        np.testing.assert_array_equal(random_vals_1, random_vals_2)


class TestDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data_basic(self, simple_dataset):
        """Test basic data preparation."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        result = sc.prepare_data(
            unit_id_col='user_id',
            treatment_col='used_smart_assistant',
            outcome_pre_col='filed_2023',
            outcome_post_col='filed_2024',
            predictor_cols=['filed_2023', 'age', 'tech_savviness']
        )
        
        assert 'X_treated' in result
        assert 'X_donors' in result
        assert 'predictor_cols' in result
        assert result['n_treated'] > 0
        assert result['n_donors'] > 0
        assert result['n_treated'] + result['n_donors'] == len(simple_dataset)
    
    def test_prepare_data_default_predictors(self, simple_dataset):
        """Test data preparation with default predictors."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        result = sc.prepare_data()
        
        # Should use default predictors that exist in the data
        assert len(result['predictor_cols']) > 0
        assert all(col in simple_dataset.columns for col in result['predictor_cols'])
    
    def test_prepare_data_missing_predictors(self, simple_dataset):
        """Test behavior with missing predictor columns."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Try with non-existent predictors
        with pytest.raises(ValueError, match="No predictor columns found"):
            sc.prepare_data(predictor_cols=['nonexistent_col'])
    
    def test_prepare_data_no_treated_units(self, simple_dataset):
        """Test behavior when no treated units exist."""
        # Set all treatment to 0
        simple_dataset['used_smart_assistant'] = 0
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        with pytest.raises(ValueError, match="No treated units found"):
            sc.prepare_data()
    
    def test_prepare_data_no_control_units(self, simple_dataset):
        """Test behavior when no control units exist."""
        # Set all treatment to 1
        simple_dataset['used_smart_assistant'] = 1
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        with pytest.raises(ValueError, match="No control units found"):
            sc.prepare_data()
    
    def test_prepare_data_handles_missing_values(self, edge_case_dataset):
        """Test handling of missing values."""
        sc = SyntheticControl(edge_case_dataset, random_state=42)
        
        with pytest.warns(UserWarning, match="Missing values detected"):
            result = sc.prepare_data(
                predictor_cols=['predictor_1', 'predictor_2']
            )
        
        # Should complete without error
        assert 'X_treated' in result
        assert 'X_donors' in result
        
        # No NaN values should remain
        assert not np.any(np.isnan(result['X_treated']))
        assert not np.any(np.isnan(result['X_donors']))


class TestWeightOptimization:
    """Test weight optimization functionality."""
    
    def test_optimize_weights_basic(self, simple_dataset):
        """Test basic weight optimization."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Prepare some test data
        treated_char = np.array([1.0, 0.5, 2.0])
        donor_chars = np.array([[1.1, 0.4, 2.1], 
                               [0.9, 0.6, 1.9], 
                               [2.0, 1.0, 1.0]])
        
        weights = sc._optimize_weights(treated_char, donor_chars)
        
        # Weights should sum to 1
        assert abs(weights.sum() - 1.0) < 1e-6
        
        # All weights should be non-negative
        assert all(w >= 0 for w in weights)
        
        # Should have one weight per donor
        assert len(weights) == len(donor_chars)
    
    def test_optimize_weights_perfect_match(self, simple_dataset):
        """Test weight optimization when perfect match exists."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Create case where first donor is perfect match
        treated_char = np.array([1.0, 0.5, 2.0])
        donor_chars = np.array([[1.0, 0.5, 2.0],  # Perfect match
                               [0.9, 0.6, 1.9], 
                               [2.0, 1.0, 1.0]])
        
        weights = sc._optimize_weights(treated_char, donor_chars)
        
        # First weight should be close to 1, others close to 0
        assert weights[0] > 0.9
        assert all(w < 0.1 for w in weights[1:])
    
    def test_optimize_weights_single_donor(self, simple_dataset):
        """Test weight optimization with single donor."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        treated_char = np.array([1.0, 0.5])
        donor_chars = np.array([[1.1, 0.4]])
        
        weights = sc._optimize_weights(treated_char, donor_chars)
        
        # Should return weight of 1 for single donor
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6


class TestSyntheticControlConstruction:
    """Test synthetic control construction."""
    
    def test_construct_synthetic_controls_basic(self, simple_dataset):
        """Test basic synthetic control construction."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        results = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age', 'tech_savviness']
        )
        
        # Check result structure
        assert 'individual_results' in results
        assert 'average_treatment_effect' in results
        assert 'ate_std_error' in results
        assert 'median_treatment_effect' in results
        
        # Check individual results
        individual_df = results['individual_results']
        required_cols = ['unit_id', 'actual_pre', 'actual_post', 'synthetic_pre', 
                        'synthetic_post', 'treatment_effect', 'pre_treatment_error']
        
        for col in required_cols:
            assert col in individual_df.columns
        
        # Check that we have results for all treated units
        n_treated = simple_dataset['used_smart_assistant'].sum()
        assert len(individual_df) == n_treated
    
    def test_construct_synthetic_controls_with_standardization(self, simple_dataset):
        """Test synthetic control construction with standardization."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        results_std = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age', 'tech_savviness'],
            standardize=True
        )
        
        results_no_std = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age', 'tech_savviness'],
            standardize=False
        )
        
        # Results might differ due to standardization
        assert 'average_treatment_effect' in results_std
        assert 'average_treatment_effect' in results_no_std
    
    def test_construct_synthetic_controls_stores_weights(self, simple_dataset):
        """Test that synthetic control construction stores weights."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        results = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age']
        )
        
        # Should store weights for each treated unit
        assert len(sc.synthetic_weights) > 0
        
        # Each weight vector should sum to 1
        for unit_id, weights in sc.synthetic_weights.items():
            assert abs(weights.sum() - 1.0) < 1e-6
            assert all(w >= 0 for w in weights)
    
    def test_construct_synthetic_controls_quality_metrics(self, simple_dataset):
        """Test quality metrics calculation."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        results = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age', 'tech_savviness']
        )
        
        # Quality metrics should be non-negative
        assert results['average_pre_treatment_error'] >= 0
        assert results['weight_concentration'] >= 0
        
        # Weight concentration should be reasonable (between 0 and 1)
        assert 0 <= results['weight_concentration'] <= 1


class TestPlaceboTesting:
    """Test placebo testing functionality."""
    
    def test_estimate_statistical_significance_basic(self, simple_dataset):
        """Test basic placebo testing."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # First run main analysis
        results = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age']
        )
        
        # Run placebo tests (small number for speed)
        placebo_results = sc.estimate_statistical_significance(n_placebo=5)
        
        assert 'observed_ate' in placebo_results
        assert 'placebo_effects' in placebo_results
        assert 'p_value' in placebo_results
        assert 'n_placebo_tests' in placebo_results
        
        # P-value should be between 0 and 1
        assert 0 <= placebo_results['p_value'] <= 1
        
        # Should have run some placebo tests
        assert placebo_results['n_placebo_tests'] > 0
    
    def test_estimate_statistical_significance_without_main_analysis(self, simple_dataset):
        """Test placebo testing without running main analysis first."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        with pytest.raises(ValueError, match="Must run construct_synthetic_controls"):
            sc.estimate_statistical_significance(n_placebo=5)
    
    def test_estimate_statistical_significance_empty_results(self, simple_dataset):
        """Test behavior when placebo tests encounter errors."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Run main analysis
        results = sc.construct_synthetic_controls()
        
        # Test with very small number of placebo tests to handle potential failures gracefully
        placebo_results = sc.estimate_statistical_significance(n_placebo=2)
        
        # Should handle gracefully and return valid results
        assert 'p_value' in placebo_results
        assert 'placebo_effects' in placebo_results
        assert 'observed_ate' in placebo_results
        assert 0 <= placebo_results['p_value'] <= 1
        assert placebo_results['observed_ate'] == results['average_treatment_effect']


class TestVisualization:
    """Test visualization functionality."""
    
    def test_plot_treatment_effects_basic(self, simple_dataset):
        """Test basic plotting functionality."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Run analysis first
        results = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age']
        )
        
        # Create plot
        fig = sc.plot_treatment_effects(figsize=(12, 8))
        
        assert fig is not None
        assert len(fig.axes) == 6  # Should have 6 subplots
        
        # Clean up
        plt.close(fig)
    
    def test_plot_treatment_effects_without_results(self, simple_dataset):
        """Test plotting without running analysis first."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        with pytest.raises(ValueError, match="Must run construct_synthetic_controls"):
            sc.plot_treatment_effects()


class TestSummaryReport:
    """Test summary report generation."""
    
    def test_generate_summary_report_basic(self, simple_dataset):
        """Test basic summary report generation."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Run analysis first
        results = sc.construct_synthetic_controls()
        
        report = sc.generate_summary_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "SYNTHETIC CONTROL ANALYSIS SUMMARY" in report
        assert "TREATMENT EFFECT ESTIMATES" in report
        assert "RECOMMENDATIONS" in report
    
    def test_generate_summary_report_without_results(self, simple_dataset):
        """Test summary report generation without results."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        report = sc.generate_summary_report()
        
        assert "No synthetic control results available" in report


class TestConvenienceFunction:
    """Test convenience function for loading and analyzing data."""
    
    def test_load_and_analyze_synthetic_control_csv(self, simple_dataset, tmp_path):
        """Test convenience function with CSV file."""
        # Save dataset to temporary CSV
        csv_path = tmp_path / "test_data.csv"
        simple_dataset.to_csv(csv_path, index=False)
        
        # Test the convenience function
        analysis = load_and_analyze_synthetic_control(
            str(csv_path),
            predictor_cols=['filed_2023', 'age']
        )
        
        assert 'synthetic_control' in analysis
        assert 'results' in analysis
        assert 'summary_report' in analysis
        
        # Check that analysis ran successfully
        assert analysis['results']['average_treatment_effect'] is not None
    
    def test_load_and_analyze_synthetic_control_unsupported_format(self, tmp_path):
        """Test convenience function with unsupported file format."""
        txt_path = tmp_path / "test_data.txt"
        txt_path.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_and_analyze_synthetic_control(str(txt_path))


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_small_sample_size(self, edge_case_dataset):
        """Test behavior with small sample size."""
        sc = SyntheticControl(edge_case_dataset, random_state=42)
        
        # Should complete without error despite small sample
        results = sc.construct_synthetic_controls(
            predictor_cols=['predictor_1', 'predictor_2']
        )
        
        assert 'average_treatment_effect' in results
        assert results['average_treatment_effect'] is not None
    
    def test_binary_outcome_handling(self, simple_dataset):
        """Test proper handling of binary outcomes."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Ensure outcomes are boolean
        simple_dataset['filed_2023'] = simple_dataset['filed_2023'].astype(bool)
        simple_dataset['filed_2024'] = simple_dataset['filed_2024'].astype(bool)
        
        # Should handle boolean outcomes correctly
        results = sc.construct_synthetic_controls()
        
        assert 'average_treatment_effect' in results
        # Treatment effects should be reasonable for binary outcomes
        assert -1 <= results['average_treatment_effect'] <= 1
    
    def test_extreme_weight_concentration(self, simple_dataset):
        """Test behavior when weights are highly concentrated."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Create scenario likely to produce concentrated weights
        # Make one control unit very similar to all treated units
        treated_mask = simple_dataset['used_smart_assistant'] == 1
        control_mask = simple_dataset['used_smart_assistant'] == 0
        
        if control_mask.sum() > 0:
            # Set first control unit to be very similar to treated units
            control_idx = simple_dataset[control_mask].index[0]
            treated_means = simple_dataset[treated_mask][['age', 'tech_savviness']].mean()
            
            simple_dataset.loc[control_idx, 'age'] = int(treated_means['age'])
            simple_dataset.loc[control_idx, 'tech_savviness'] = float(treated_means['tech_savviness'])
        
        results = sc.construct_synthetic_controls(
            predictor_cols=['age', 'tech_savviness']
        )
        
        # Should complete even with potentially concentrated weights
        assert 'weight_concentration' in results
        assert results['weight_concentration'] >= 0


class TestIntegration:
    """Integration tests for synthetic control workflow."""
    
    def test_full_workflow(self, simple_dataset):
        """Test complete synthetic control workflow."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # 1. Construct synthetic controls
        results = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age', 'tech_savviness']
        )
        
        # 2. Run placebo tests
        placebo_results = sc.estimate_statistical_significance(n_placebo=5)
        
        # 3. Generate visualization
        fig = sc.plot_treatment_effects()
        plt.close(fig)
        
        # 4. Generate report
        report = sc.generate_summary_report()
        
        # Verify all components worked
        assert results['average_treatment_effect'] is not None
        assert placebo_results['p_value'] is not None
        assert len(report) > 0
        assert "SYNTHETIC CONTROL ANALYSIS SUMMARY" in report
    
    def test_reproducibility(self, simple_dataset):
        """Test that results are reproducible with same random seed."""
        # Run analysis twice with same seed
        sc1 = SyntheticControl(simple_dataset, random_state=42)
        results1 = sc1.construct_synthetic_controls()
        
        sc2 = SyntheticControl(simple_dataset, random_state=42)
        results2 = sc2.construct_synthetic_controls()
        
        # Results should be identical
        assert results1['average_treatment_effect'] == results2['average_treatment_effect']
        assert results1['ate_std_error'] == results2['ate_std_error']
    
    def test_different_parameters_produce_different_results(self, simple_dataset):
        """Test that different parameters produce different results."""
        sc = SyntheticControl(simple_dataset, random_state=42)
        
        # Run with different predictor sets
        results1 = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023']
        )
        
        results2 = sc.construct_synthetic_controls(
            predictor_cols=['filed_2023', 'age', 'tech_savviness']
        )
        
        # Results should be different (though possibly not by much)
        # At minimum, pre-treatment errors should differ
        assert (results1['average_pre_treatment_error'] != 
                results2['average_pre_treatment_error']) 