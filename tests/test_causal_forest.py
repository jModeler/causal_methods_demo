"""
Unit tests for Causal Forest implementation.

This module contains comprehensive tests for the CausalForest class,
including tests for model fitting, treatment effect estimation,
feature importance calculation, and visualization methods.
"""

import warnings
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.causal_methods.causal_forest import CausalForest


class TestCausalForestInitialization:
    """Test CausalForest initialization and basic properties."""

    def test_initialization_with_valid_data(self, sample_dataset):
        """Test that CausalForest initializes correctly with valid data."""
        cf = CausalForest(sample_dataset, random_state=42)

        assert cf.random_state == 42
        assert cf.n_estimators == 100  # default value
        assert cf.max_depth is None  # default value
        assert cf.is_fitted is False
        assert isinstance(cf.data, pd.DataFrame)
        assert len(cf.data) == len(sample_dataset)

    def test_initialization_with_custom_params(self, sample_dataset):
        """Test initialization with custom parameters."""
        cf = CausalForest(
            sample_dataset,
            random_state=123,
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=3
        )

        assert cf.random_state == 123
        assert cf.n_estimators == 50
        assert cf.max_depth == 5
        assert cf.min_samples_split == 10
        assert cf.min_samples_leaf == 3

    def test_data_copying(self, sample_dataset):
        """Test that data is copied during initialization."""
        original_data = sample_dataset.copy()
        cf = CausalForest(sample_dataset)

        # Modify original data
        sample_dataset.loc[0, 'age'] = 999

        # Check that CausalForest data is unchanged
        assert cf.data.loc[0, 'age'] != 999
        assert cf.data.loc[0, 'age'] == original_data.loc[0, 'age']


class TestCausalForestFitting:
    """Test model fitting functionality."""

    def test_fit_causal_forest_basic(self, sample_dataset):
        """Test basic causal forest fitting."""
        cf = CausalForest(sample_dataset, random_state=42)

        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness', 'sessions_2023'],
            test_size=0.3
        )

        assert cf.is_fitted is True
        assert isinstance(performance, dict)
        assert 'implementation' in performance
        assert 'train_ate' in performance
        assert 'test_ate' in performance
        assert 'heterogeneity_measure' in performance

        # Check treatment effects are calculated
        assert hasattr(cf, 'treatment_effects')
        assert 'ate' in cf.treatment_effects
        assert 'individual_effects' in cf.treatment_effects
        assert 'p_value' in cf.treatment_effects

    def test_fit_with_automatic_covariate_selection(self, sample_dataset):
        """Test fitting with automatic covariate selection."""
        cf = CausalForest(sample_dataset, random_state=42)

        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=None  # Should auto-select
        )

        assert cf.is_fitted is True
        assert len(cf.covariate_cols) > 0
        assert 'filed_2024' not in cf.covariate_cols
        assert 'used_smart_assistant' not in cf.covariate_cols
        assert 'user_id' not in cf.covariate_cols

    def test_fit_with_binary_outcome(self, sample_dataset):
        """Test fitting with binary outcome variable."""
        cf = CausalForest(sample_dataset, random_state=42)

        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',  # Binary outcome
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness']
        )

        assert cf.is_fitted is True
        assert cf.is_binary_outcome is True
        assert isinstance(performance['train_ate'], int | float)

    def test_fit_with_continuous_outcome(self, sample_dataset):
        """Test fitting with continuous outcome variable."""
        # Create a continuous outcome
        sample_dataset['continuous_outcome'] = (
            sample_dataset['sessions_2023'] +
            np.random.normal(0, 1, len(sample_dataset))
        )

        cf = CausalForest(sample_dataset, random_state=42)

        performance = cf.fit_causal_forest(
            outcome_col='continuous_outcome',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness']
        )

        assert cf.is_fitted is True
        assert isinstance(performance['train_ate'], int | float)

    def test_fit_handles_insufficient_treatment_variation(self, sample_dataset):
        """Test handling of data with insufficient treatment variation."""
        # Create data with all treated units
        sample_dataset['no_variation_treatment'] = 1

        cf = CausalForest(sample_dataset, random_state=42)

        with pytest.raises(ValueError, match="Need both treated and control units"):
            cf.fit_causal_forest(
                outcome_col='filed_2024',
                treatment_col='no_variation_treatment',
                covariate_cols=['age', 'tech_savviness']
            )

    def test_fit_with_missing_columns(self, sample_dataset):
        """Test error handling for missing columns."""
        cf = CausalForest(sample_dataset, random_state=42)

        with pytest.raises(KeyError):
            cf.fit_causal_forest(
                outcome_col='nonexistent_outcome',
                treatment_col='used_smart_assistant',
                covariate_cols=['age', 'tech_savviness']
            )

    def test_feature_importance_calculation(self, sample_dataset):
        """Test that feature importance is calculated correctly."""
        cf = CausalForest(sample_dataset, random_state=42)

        cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness', 'sessions_2023']
        )

        assert hasattr(cf, 'feature_importance')
        assert 'importances' in cf.feature_importance
        assert 'feature_names' in cf.feature_importance
        assert 'sorted_indices' in cf.feature_importance

        # Check dimensions match
        assert len(cf.feature_importance['importances']) == len(cf.covariate_cols)
        assert len(cf.feature_importance['feature_names']) == len(cf.covariate_cols)
        assert len(cf.feature_importance['sorted_indices']) == len(cf.covariate_cols)

        # Check values are non-negative
        assert all(imp >= 0 for imp in cf.feature_importance['importances'])


class TestTreatmentEffectEstimation:
    """Test treatment effect estimation methods."""

    def test_treatment_effect_structure(self, fitted_causal_forest):
        """Test that treatment effects have correct structure."""
        cf = fitted_causal_forest
        te = cf.treatment_effects

        required_keys = [
            'individual_effects', 'ate', 'ate_se', 'ate_ci_lower',
            'ate_ci_upper', 'heterogeneity_std', 'p_value'
        ]

        for key in required_keys:
            assert key in te, f"Missing key: {key}"

        # Check types and shapes
        assert isinstance(te['individual_effects'], np.ndarray)
        assert len(te['individual_effects']) > 0
        assert isinstance(te['ate'], int | float)
        assert isinstance(te['ate_se'], int | float)
        assert te['ate_se'] >= 0
        assert te['ate_ci_lower'] <= te['ate_ci_upper']

    def test_confidence_intervals(self, fitted_causal_forest):
        """Test confidence interval calculation."""
        cf = fitted_causal_forest
        te = cf.treatment_effects

        # Check that confidence intervals are reasonable
        assert te['ate_ci_lower'] <= te['ate'] <= te['ate_ci_upper']

        # Check individual CIs exist and have right shape
        assert 'individual_ci_lower' in te
        assert 'individual_ci_upper' in te
        assert len(te['individual_ci_lower']) == len(te['individual_effects'])
        assert len(te['individual_ci_upper']) == len(te['individual_effects'])

    def test_heterogeneity_measures(self, fitted_causal_forest):
        """Test heterogeneity measurement."""
        cf = fitted_causal_forest
        te = cf.treatment_effects
        performance = cf.model_performance

        assert te['heterogeneity_std'] >= 0
        assert 'heterogeneity_measure' in performance
        assert performance['heterogeneity_measure'] >= 0


class TestConditionalEffects:
    """Test conditional treatment effect estimation."""

    def test_estimate_conditional_effects_basic(self, fitted_causal_forest):
        """Test basic conditional effect estimation."""
        cf = fitted_causal_forest

        feature_values = {'age': 35, 'tech_savviness': 7}
        result = cf.estimate_conditional_effects(feature_values)

        assert 'conditional_treatment_effect' in result
        assert 'feature_values' in result
        assert 'unspecified_features_used_median' in result

        assert isinstance(result['conditional_treatment_effect'], int | float)
        assert result['feature_values'] == feature_values

    def test_conditional_effects_with_all_features(self, fitted_causal_forest):
        """Test conditional effects when all features are specified."""
        cf = fitted_causal_forest

        # Specify all covariates
        feature_values = dict.fromkeys(cf.covariate_cols, 1.0)
        result = cf.estimate_conditional_effects(feature_values)

        assert len(result['unspecified_features_used_median']) == 0
        assert isinstance(result['conditional_treatment_effect'], int | float)

    def test_conditional_effects_before_fitting(self, sample_dataset):
        """Test error when estimating conditional effects before fitting."""
        cf = CausalForest(sample_dataset, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted"):
            cf.estimate_conditional_effects({'age': 35})


class TestVisualization:
    """Test visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Set up matplotlib for testing."""
        plt.ioff()  # Turn off interactive mode
        yield
        plt.close('all')  # Clean up plots

    def test_plot_treatment_effect_distribution(self, fitted_causal_forest):
        """Test treatment effect distribution plotting."""
        cf = fitted_causal_forest

        # Should not raise an error
        try:
            cf.plot_treatment_effect_distribution(bins=20, figsize=(10, 6))
            # If we get here without exception, the test passes
            assert True
        except Exception as e:
            pytest.fail(f"plot_treatment_effect_distribution raised an exception: {e}")

    def test_plot_feature_importance(self, fitted_causal_forest):
        """Test feature importance plotting."""
        cf = fitted_causal_forest

        # Should not raise an error
        try:
            cf.plot_feature_importance(top_n=5, figsize=(8, 6))
            # If we get here without exception, the test passes
            assert True
        except Exception as e:
            pytest.fail(f"plot_feature_importance raised an exception: {e}")

    def test_plot_before_fitting_raises_error(self, sample_dataset):
        """Test that plotting before fitting raises appropriate errors."""
        cf = CausalForest(sample_dataset, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted"):
            cf.plot_treatment_effect_distribution()

        with pytest.raises(ValueError, match="Model must be fitted"):
            cf.plot_feature_importance()


class TestSummaryReport:
    """Test summary report generation."""

    def test_generate_summary_report_basic(self, fitted_causal_forest):
        """Test basic summary report generation."""
        cf = fitted_causal_forest

        report = cf.generate_summary_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert "CAUSAL FOREST ANALYSIS REPORT" in report
        assert "Average Treatment Effect" in report
        assert "HETEROGENEITY ANALYSIS" in report
        assert "BUSINESS RECOMMENDATIONS" in report

    def test_summary_report_contains_key_metrics(self, fitted_causal_forest):
        """Test that summary report contains key metrics."""
        cf = fitted_causal_forest

        report = cf.generate_summary_report()

        # Check for key sections
        required_sections = [
            "MODEL INFORMATION",
            "TREATMENT EFFECT RESULTS",
            "HETEROGENEITY ANALYSIS",
            "TOP FEATURES",
            "BUSINESS RECOMMENDATIONS",
            "MODEL PERFORMANCE"
        ]

        for section in required_sections:
            assert section in report

    def test_summary_report_before_fitting(self, sample_dataset):
        """Test error when generating report before fitting."""
        cf = CausalForest(sample_dataset, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted"):
            cf.generate_summary_report()


class TestEconMLIntegration:
    """Test EconML integration functionality."""

    @patch('src.causal_methods.causal_forest.ECONML_AVAILABLE', True)
    def test_econml_available_path(self, sample_dataset):
        """Test behavior when EconML is available."""
        with patch('src.causal_methods.causal_forest.CausalForestDML') as mock_econml:
            # Mock the EconML CausalForestDML
            mock_model = MagicMock()
            mock_model.ate.return_value = 0.05
            mock_model.effect.return_value = np.array([0.03, 0.04, 0.06, 0.05])
            mock_econml.return_value = mock_model

            cf = CausalForest(sample_dataset, random_state=42)

            performance = cf.fit_causal_forest(
                outcome_col='filed_2024',
                treatment_col='used_smart_assistant',
                covariate_cols=['age', 'tech_savviness']
            )

            assert performance['implementation'] == 'EconML CausalForestDML'
            assert mock_econml.called

    @patch('src.causal_methods.causal_forest.ECONML_AVAILABLE', False)
    def test_econml_unavailable_fallback(self, sample_dataset):
        """Test fallback behavior when EconML is unavailable."""
        cf = CausalForest(sample_dataset, random_state=42)

        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness']
        )

        assert performance['implementation'] == 'Simple T-learner'


class TestRobustness:
    """Test robustness to edge cases and problematic data."""

    def test_robustness_to_small_sample(self, sample_dataset):
        """Test behavior with very small samples."""
        small_data = sample_dataset.head(20).copy()
        cf = CausalForest(small_data, random_state=42, min_samples_leaf=1)

        # Should still work with small samples
        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness'],
            test_size=0.3
        )

        assert cf.is_fitted is True
        assert isinstance(performance['train_ate'], int | float)

    def test_robustness_to_constant_features(self, sample_dataset):
        """Test behavior with constant features."""
        sample_dataset['constant_feature'] = 1.0

        cf = CausalForest(sample_dataset, random_state=42)

        # Should handle constant features gracefully
        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'constant_feature']
        )

        assert cf.is_fitted is True

    def test_robustness_to_outliers(self, sample_dataset):
        """Test behavior with outliers in the data."""
        # Add extreme outliers
        sample_dataset.loc[0, 'age'] = 999
        sample_dataset.loc[1, 'tech_savviness'] = 1e6

        cf = CausalForest(sample_dataset, random_state=42)

        # Should handle outliers gracefully
        performance = cf.fit_causal_forest(
            outcome_col='filed_2024',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness', 'sessions_2023']
        )

        assert cf.is_fitted is True
        assert not np.isnan(performance['train_ate'])
        assert not np.isnan(performance['test_ate'])

    def test_robustness_to_single_class_outcome(self, sample_dataset):
        """Test behavior when outcome has only one class in splits."""
        # Create data where some splits might have only one class
        sample_dataset['extreme_outcome'] = 0
        sample_dataset.loc[:5, 'extreme_outcome'] = 1

        cf = CausalForest(sample_dataset, random_state=42, min_samples_leaf=1)

        # Should handle single-class splits gracefully
        performance = cf.fit_causal_forest(
            outcome_col='extreme_outcome',
            treatment_col='used_smart_assistant',
            covariate_cols=['age', 'tech_savviness']
        )

        assert cf.is_fitted is True


class TestWarningHandling:
    """Test proper warning handling."""

    def test_warning_suppression(self, sample_dataset):
        """Test that warnings are properly handled."""
        cf = CausalForest(sample_dataset, random_state=42)

        # This should not produce warnings in the output
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            cf.fit_causal_forest(
                outcome_col='filed_2024',
                treatment_col='used_smart_assistant',
                covariate_cols=['age', 'tech_savviness']
            )

            # Should not have critical warnings
            critical_warnings = [warn for warn in w
                               if "error" in str(warn.message).lower()]
            assert len(critical_warnings) == 0


@pytest.fixture
def fitted_causal_forest(sample_dataset):
    """Fixture that provides a fitted causal forest model."""
    cf = CausalForest(sample_dataset, random_state=42)

    cf.fit_causal_forest(
        outcome_col='filed_2024',
        treatment_col='used_smart_assistant',
        covariate_cols=['age', 'tech_savviness', 'sessions_2023'],
        test_size=0.3
    )

    return cf
