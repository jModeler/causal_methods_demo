"""
Tests for Double Machine Learning (DML) implementation.

Tests cover:
- DML initialization and basic functionality
- Treatment effect estimation with different models
- Multiple outcome analysis
- Model comparison functionality
- Visualization methods
- Error handling and edge cases
- Integration with data simulation
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.causal_methods.dml import DoubleMachineLearning, load_and_analyze_dml
from src.data_simulation import generate_and_save_data

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 200

    # Generate covariates
    age = np.random.normal(40, 10, n)
    income = np.random.uniform(30000, 100000, n)
    tech_savviness = np.random.uniform(0, 1, n)

    # Generate treatment with selection bias
    treatment_prob = 0.3 + 0.3 * (tech_savviness > 0.5) + 0.2 * (age < 35)
    treatment = np.random.binomial(1, treatment_prob)

    # Generate outcome with treatment effect
    outcome_continuous = (
        50
        + 0.5 * age
        + 0.0001 * income
        + 10 * tech_savviness
        + 5 * treatment
        + np.random.normal(0, 5, n)
    )
    outcome_binary = (outcome_continuous > np.median(outcome_continuous)).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "tech_savviness": tech_savviness,
            "treatment": treatment,
            "outcome_continuous": outcome_continuous,
            "outcome_binary": outcome_binary,
        }
    )


@pytest.fixture
def tax_data():
    """Generate tax software data for testing."""
    return generate_and_save_data(
        output_path=None,  # Return dataframe instead of saving
        n_users=300,
        config_path="config/simulation_config.yaml",
    )


class TestDMLInitialization:
    """Test DML initialization and basic properties."""

    def test_initialization(self, sample_data):
        """Test basic initialization."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        assert isinstance(dml.data, pd.DataFrame)
        assert len(dml.data) == len(sample_data)
        assert dml.random_state == 42
        assert not dml.fitted
        assert dml.treatment_effects == {}
        assert dml.residuals == {}

    def test_initialization_with_missing_data(self, sample_data):
        """Test initialization with missing data."""
        # Add some missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:10, "age"] = np.nan

        dml = DoubleMachineLearning(data_with_missing, random_state=42)
        assert isinstance(dml.data, pd.DataFrame)
        # Should handle missing data during analysis, not initialization

    def test_get_available_models(self, sample_data):
        """Test getting available models."""
        dml = DoubleMachineLearning(sample_data)

        # Test regression models
        reg_models = dml.get_available_models("regression")
        expected_reg = [
            "random_forest",
            "gradient_boosting",
            "linear_regression",
            "ridge",
        ]
        assert all(model in reg_models for model in expected_reg)

        # Test classification models
        clf_models = dml.get_available_models("classification")
        expected_clf = [
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "ridge_classifier",
        ]
        assert all(model in clf_models for model in expected_clf)

    def test_invalid_model_type(self, sample_data):
        """Test invalid model type."""
        dml = DoubleMachineLearning(sample_data)

        with pytest.raises(ValueError, match="model_type must be"):
            dml.get_available_models("invalid_type")


class TestDMLTreatmentEffects:
    """Test treatment effect estimation."""

    def test_estimate_treatment_effects_continuous_outcome(self, sample_data):
        """Test treatment effect estimation with continuous outcome."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        results = dml.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income", "tech_savviness"],
            outcome_model="random_forest",
            treatment_model="random_forest",
            n_folds=2,
        )

        # Check result structure
        required_keys = ["ate", "se", "ci_lower", "ci_upper", "p_value", "n_samples"]
        assert all(key in results for key in required_keys)

        # Check values are reasonable
        assert isinstance(results["ate"], int | float)
        assert isinstance(results["se"], int | float)
        assert results["se"] >= 0
        assert results["ci_lower"] <= results["ci_upper"]
        assert 0 <= results["p_value"] <= 1
        assert results["n_samples"] > 0

        # Check DML object state
        assert dml.fitted
        assert "outcome_continuous" in dml.treatment_effects
        assert "outcome_continuous" in dml.residuals

    def test_estimate_treatment_effects_binary_outcome(self, sample_data):
        """Test treatment effect estimation with binary outcome."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        results = dml.estimate_treatment_effects(
            outcome_col="outcome_binary",
            treatment_col="treatment",
            covariates=["age", "income", "tech_savviness"],
            outcome_model="random_forest",
            treatment_model="logistic_regression",
            n_folds=2,
        )

        # Check results
        assert isinstance(results["ate"], int | float)
        assert results["outcome_is_binary"]

        # Check fold performance includes classification metrics
        for fold_data in results["fold_performance"]:
            assert "accuracy" in fold_data["outcome_performance"]

    def test_different_models(self, sample_data):
        """Test different ML model combinations."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        # Test different outcome models
        for outcome_model in [
            "random_forest",
            "gradient_boosting",
            "linear_regression",
        ]:
            results = dml.estimate_treatment_effects(
                outcome_col="outcome_continuous",
                treatment_col="treatment",
                covariates=["age", "income"],
                outcome_model=outcome_model,
                treatment_model="random_forest",
                n_folds=2,
            )
            assert isinstance(results["ate"], int | float)

    def test_different_fold_numbers(self, sample_data):
        """Test different numbers of cross-fitting folds."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        for n_folds in [2, 3, 5]:
            results = dml.estimate_treatment_effects(
                outcome_col="outcome_continuous",
                treatment_col="treatment",
                covariates=["age", "income"],
                n_folds=n_folds,
            )
            assert results["n_folds"] == n_folds
            assert len(results["fold_performance"]) == n_folds

    def test_feature_scaling(self, sample_data):
        """Test with and without feature scaling."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        # Test with scaling
        results_scaled = dml.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income", "tech_savviness"],
            scale_features=True,
        )

        # Test without scaling
        results_unscaled = dml.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income", "tech_savviness"],
            scale_features=False,
        )

        # Both should work
        assert isinstance(results_scaled["ate"], int | float)
        assert isinstance(results_unscaled["ate"], int | float)

    def test_estimate_multiple_outcomes(self, sample_data):
        """Test estimating effects for multiple outcomes."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        results = dml.estimate_multiple_outcomes(
            outcome_cols=["outcome_continuous", "outcome_binary"],
            treatment_col="treatment",
            covariates=["age", "income"],
            n_folds=2,
        )

        assert len(results) == 2
        assert "outcome_continuous" in results
        assert "outcome_binary" in results

        for _outcome, result in results.items():
            assert "ate" in result
            assert "se" in result


class TestDMLModelComparison:
    """Test model comparison functionality."""

    def test_compare_models_continuous(self, sample_data):
        """Test model comparison for continuous outcomes."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        comparison = dml.compare_models(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income"],
            n_folds=2,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0
        assert "outcome_model" in comparison.columns
        assert "treatment_model" in comparison.columns
        assert "ate" in comparison.columns
        assert "p_value" in comparison.columns

    def test_compare_models_binary(self, sample_data):
        """Test model comparison for binary outcomes."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        comparison = dml.compare_models(
            outcome_col="outcome_binary",
            treatment_col="treatment",
            covariates=["age", "income"],
            n_folds=2,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0


class TestDMLVisualization:
    """Test visualization methods."""

    def test_plot_residuals(self, sample_data):
        """Test residual plotting."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        # First estimate effects
        dml.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        # Test plotting
        fig = dml.plot_residuals("outcome_continuous")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)

    def test_plot_treatment_effects(self, sample_data):
        """Test treatment effect plotting."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        # Estimate effects for multiple outcomes
        dml.estimate_multiple_outcomes(
            outcome_cols=["outcome_continuous", "outcome_binary"],
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        # Test plotting
        fig = dml.plot_treatment_effects()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_without_results(self, sample_data):
        """Test plotting methods without results."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        with pytest.raises(ValueError):
            dml.plot_residuals("outcome_continuous")

        with pytest.raises(ValueError):
            dml.plot_treatment_effects()


class TestDMLReporting:
    """Test reporting functionality."""

    def test_generate_summary_report_single(self, sample_data):
        """Test summary report generation for single outcome."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        dml.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        report = dml.generate_summary_report("outcome_continuous")
        assert isinstance(report, str)
        assert "outcome_continuous" in report
        assert "Average Treatment Effect" in report
        assert "Standard Error" in report

    def test_generate_summary_report_multiple(self, sample_data):
        """Test summary report generation for multiple outcomes."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        dml.estimate_multiple_outcomes(
            outcome_cols=["outcome_continuous", "outcome_binary"],
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        report = dml.generate_summary_report()
        assert isinstance(report, str)
        assert "outcome_continuous" in report
        assert "outcome_binary" in report

    def test_generate_summary_report_no_results(self, sample_data):
        """Test summary report without results."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        report = dml.generate_summary_report()
        assert "No treatment effects estimated" in report


class TestDMLErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_columns(self, sample_data):
        """Test handling of missing columns."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        with pytest.raises(KeyError):
            dml.estimate_treatment_effects(
                outcome_col="nonexistent_outcome",
                treatment_col="treatment",
                covariates=["age", "income"],
            )

    def test_invalid_model_names(self, sample_data):
        """Test invalid model names."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        with pytest.raises(ValueError):
            dml.estimate_treatment_effects(
                outcome_col="outcome_continuous",
                treatment_col="treatment",
                covariates=["age", "income"],
                outcome_model="invalid_model",
            )

    def test_insufficient_data(self):
        """Test with very small dataset."""
        small_data = pd.DataFrame(
            {"outcome": [1, 2], "treatment": [0, 1], "covariate": [1, 2]}
        )

        dml = DoubleMachineLearning(small_data, random_state=42)

        # Should handle gracefully, might produce warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = dml.estimate_treatment_effects(
                outcome_col="outcome",
                treatment_col="treatment",
                covariates=["covariate"],
                n_folds=2,
            )
            # Results might be unstable but shouldn't crash
            assert "ate" in results

    def test_no_treatment_variation(self, sample_data):
        """Test with no treatment variation."""
        no_variation_data = sample_data.copy()
        no_variation_data["treatment"] = 1  # All treated

        dml = DoubleMachineLearning(no_variation_data, random_state=42)

        # Should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = dml.estimate_treatment_effects(
                outcome_col="outcome_continuous",
                treatment_col="treatment",
                covariates=["age", "income"],
            )
            # Results will be unreliable but shouldn't crash
            assert "ate" in results

    def test_perfect_separation(self, sample_data):
        """Test with perfect treatment-outcome correlation."""
        perfect_data = sample_data.copy()
        perfect_data["outcome_binary"] = perfect_data["treatment"]

        dml = DoubleMachineLearning(perfect_data, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = dml.estimate_treatment_effects(
                outcome_col="outcome_binary",
                treatment_col="treatment",
                covariates=["age", "income"],
            )
            assert "ate" in results


class TestDMLIntegration:
    """Test integration with tax software data and convenience functions."""

    def test_with_tax_data(self, tax_data):
        """Test DML with tax software data."""
        dml = DoubleMachineLearning(tax_data, random_state=42)

        # Select appropriate covariates
        covariates = [
            col
            for col in tax_data.columns
            if col
            not in [
                "user_id",
                "used_smart_assistant",
                "filed_2024",
                "satisfaction_2024",
            ]
            and tax_data[col].dtype in ["int64", "float64", "bool"]
        ]

        results = dml.estimate_treatment_effects(
            outcome_col="filed_2024",
            treatment_col="used_smart_assistant",
            covariates=covariates[:5],  # Use first 5 to keep test fast
            n_folds=2,
        )

        assert isinstance(results["ate"], int | float)
        assert results["n_samples"] > 0

    def test_load_and_analyze_dml_function(self, tax_data, tmp_path):
        """Test convenience function for loading and analyzing data."""
        # Save test data
        test_file = tmp_path / "test_data.csv"
        tax_data.to_csv(test_file, index=False)

        # Test convenience function
        dml = load_and_analyze_dml(
            file_path=str(test_file),
            outcome_cols="filed_2024",
            treatment_col="used_smart_assistant",
            covariates=["age", "tech_savviness"],
            n_folds=2,
        )

        assert isinstance(dml, DoubleMachineLearning)
        assert dml.fitted
        assert "filed_2024" in dml.treatment_effects

    def test_load_and_analyze_multiple_outcomes(self, tax_data, tmp_path):
        """Test convenience function with multiple outcomes."""
        # Save test data
        test_file = tmp_path / "test_data.csv"
        tax_data.to_csv(test_file, index=False)

        # Test with multiple outcomes
        dml = load_and_analyze_dml(
            file_path=str(test_file),
            outcome_cols=["filed_2024", "satisfaction_2024"],
            treatment_col="used_smart_assistant",
            covariates=["age", "tech_savviness"],
            n_folds=2,
        )

        assert isinstance(dml, DoubleMachineLearning)
        assert "filed_2024" in dml.treatment_effects
        assert "satisfaction_2024" in dml.treatment_effects


class TestDMLRobustness:
    """Test robustness and edge case handling."""

    def test_different_data_types(self, sample_data):
        """Test with different data types."""
        # Add categorical variables
        data_with_categories = sample_data.copy()
        data_with_categories["category"] = pd.Categorical(
            ["A", "B"] * (len(sample_data) // 2)
        )
        data_with_categories["bool_var"] = data_with_categories["age"] > 40

        # Convert category to numeric for DML
        data_with_categories["category_numeric"] = data_with_categories[
            "category"
        ].cat.codes

        dml = DoubleMachineLearning(data_with_categories, random_state=42)

        results = dml.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "category_numeric", "bool_var"],
        )

        assert isinstance(results["ate"], int | float)

    def test_boolean_treatment_and_outcome(self, sample_data):
        """Test with boolean treatment and outcome."""
        bool_data = sample_data.copy()
        bool_data["treatment_bool"] = bool_data["treatment"].astype(bool)
        bool_data["outcome_bool"] = bool_data["outcome_binary"].astype(bool)

        dml = DoubleMachineLearning(bool_data, random_state=42)

        results = dml.estimate_treatment_effects(
            outcome_col="outcome_bool",
            treatment_col="treatment_bool",
            covariates=["age", "income"],
        )

        assert isinstance(results["ate"], int | float)

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random seed."""
        # First run
        dml1 = DoubleMachineLearning(sample_data, random_state=42)
        results1 = dml1.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        # Second run with same seed
        dml2 = DoubleMachineLearning(sample_data, random_state=42)
        results2 = dml2.estimate_treatment_effects(
            outcome_col="outcome_continuous",
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        # Results should be very similar (allowing for small numerical differences)
        assert abs(results1["ate"] - results2["ate"]) < 1e-10

    def test_large_number_of_folds(self, sample_data):
        """Test with large number of cross-fitting folds."""
        dml = DoubleMachineLearning(sample_data, random_state=42)

        # Test with many folds (should work but might give warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = dml.estimate_treatment_effects(
                outcome_col="outcome_continuous",
                treatment_col="treatment",
                covariates=["age", "income"],
                n_folds=10,  # Many folds for small dataset
            )

        assert isinstance(results["ate"], int | float)
        assert results["n_folds"] == 10


if __name__ == "__main__":
    pytest.main([__file__])
