"""Tests for Propensity Score Matching module."""

import numpy as np
import pandas as pd
import pytest

from src.causal_methods.psm import PropensityScoreMatching, load_and_analyze_psm


class TestPropensityScoreMatching:
    """Test the PropensityScoreMatching class."""

    def test_init_with_valid_data(self, sample_dataset):
        """Test PSM class initialization with valid data."""
        psm = PropensityScoreMatching(sample_dataset)

        assert hasattr(psm, "data")
        assert len(psm.data) == len(sample_dataset)
        assert hasattr(psm, "propensity_scores")
        assert hasattr(psm, "matched_data")
        assert hasattr(psm, "balance_stats")
        assert hasattr(psm, "treatment_effects")
        assert psm.propensity_scores is None  # Not estimated yet

    def test_estimate_propensity_scores(self, sample_dataset):
        """Test propensity score estimation."""
        psm = PropensityScoreMatching(sample_dataset)

        # Test with default covariates
        results = psm.estimate_propensity_scores(treatment_col="used_smart_assistant")

        # Check that propensity scores were calculated
        assert psm.propensity_scores is not None
        assert len(psm.propensity_scores) == len(sample_dataset)
        assert "propensity_score" in psm.data.columns

        # Check results structure
        assert isinstance(results, dict)
        assert "auc_score" in results
        assert "propensity_score_range" in results
        assert "common_support" in results
        assert "feature_importance" in results

        # Check propensity score properties
        assert 0 <= results["propensity_score_range"]["min"] <= 1
        assert 0 <= results["propensity_score_range"]["max"] <= 1
        assert 0 <= results["auc_score"] <= 1

    def test_estimate_propensity_scores_custom_covariates(self, sample_dataset):
        """Test propensity score estimation with custom covariates."""
        psm = PropensityScoreMatching(sample_dataset)

        # Test with specific covariates
        custom_covariates = ["age", "tech_savviness", "filed_2023"]
        results = psm.estimate_propensity_scores(
            treatment_col="used_smart_assistant", covariates=custom_covariates
        )

        assert psm.propensity_scores is not None
        assert isinstance(results, dict)
        assert results["covariates_used"] == custom_covariates

    def test_perform_matching_nearest_neighbor(self, sample_dataset):
        """Test nearest neighbor matching."""
        psm = PropensityScoreMatching(sample_dataset)

        # Estimate propensity scores first
        psm.estimate_propensity_scores()

        # Perform matching
        results = psm.perform_matching(
            method="nearest_neighbor", caliper=0.1, replacement=False, ratio=1
        )

        # Check matching results
        assert isinstance(results, dict)
        assert "method" in results
        assert "n_treated_total" in results
        assert "n_treated_matched" in results
        assert "n_control_matched" in results
        assert "matching_rate" in results

        # Check that matched data was created
        assert psm.matched_data is not None
        assert len(psm.matched_data) > 0

        # Check matching rate is reasonable
        assert 0 <= results["matching_rate"] <= 1

    def test_perform_matching_caliper(self, sample_dataset):
        """Test caliper matching."""
        psm = PropensityScoreMatching(sample_dataset)

        # Estimate propensity scores first
        psm.estimate_propensity_scores()

        # Perform caliper matching
        results = psm.perform_matching(method="caliper", caliper=0.2)

        assert results["method"] == "caliper"
        assert psm.matched_data is not None
        assert results["caliper_used"] == 0.2

    def test_matching_without_propensity_scores(self, sample_dataset):
        """Test that matching fails without propensity scores."""
        psm = PropensityScoreMatching(sample_dataset)

        with pytest.raises(ValueError, match="Must estimate propensity scores first"):
            psm.perform_matching()

    def test_assess_balance(self, sample_dataset):
        """Test covariate balance assessment."""
        psm = PropensityScoreMatching(sample_dataset)

        # Estimate propensity scores and perform matching
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Assess balance
        balance_results = psm.assess_balance(
            covariates=["age", "tech_savviness"], treatment_col="used_smart_assistant"
        )

        # Check structure
        assert isinstance(balance_results, dict)
        assert "before_matching" in balance_results
        assert "after_matching" in balance_results
        assert "covariates_assessed" in balance_results

        # Check that balance statistics exist
        before_stats = balance_results["before_matching"]
        after_stats = balance_results["after_matching"]

        assert isinstance(before_stats, dict)
        assert isinstance(after_stats, dict)

        # Check that statistics have expected keys
        for covar in balance_results["covariates_assessed"]:
            if covar in before_stats:
                assert "standardized_mean_diff" in before_stats[covar]

    def test_estimate_treatment_effects(self, sample_dataset):
        """Test treatment effect estimation."""
        psm = PropensityScoreMatching(sample_dataset)

        # Complete matching pipeline
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Estimate treatment effects
        effects = psm.estimate_treatment_effects(
            outcome_cols="filed_2024", treatment_col="used_smart_assistant"
        )

        # Check structure
        assert isinstance(effects, dict)
        assert "filed_2024" in effects

        effect_result = effects["filed_2024"]
        expected_keys = [
            "ate",
            "ci_lower",
            "ci_upper",
            "p_value",
            "n_treated",
            "n_control",
        ]
        for key in expected_keys:
            assert key in effect_result

        # Check that confidence interval makes sense
        assert (
            effect_result["ci_lower"]
            <= effect_result["ate"]
            <= effect_result["ci_upper"]
        )

    def test_estimate_treatment_effects_multiple_outcomes(self, sample_dataset):
        """Test treatment effect estimation with multiple outcomes."""
        psm = PropensityScoreMatching(sample_dataset)

        # Complete matching pipeline
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Test with multiple outcomes
        outcome_cols = ["filed_2024", "satisfaction_2024"]
        available_outcomes = [
            col for col in outcome_cols if col in sample_dataset.columns
        ]

        if len(available_outcomes) > 1:
            effects = psm.estimate_treatment_effects(
                outcome_cols=available_outcomes, treatment_col="used_smart_assistant"
            )

            assert len(effects) == len(available_outcomes)
            for outcome in available_outcomes:
                assert outcome in effects

    def test_treatment_effects_without_matching(self, sample_dataset):
        """Test that treatment effect estimation fails without matching."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()

        with pytest.raises(ValueError, match="Must perform matching first"):
            psm.estimate_treatment_effects()

    def test_plot_propensity_distribution(self, sample_dataset):
        """Test propensity score distribution plotting."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()

        # Test plotting
        fig = psm.plot_propensity_distribution()
        assert fig is not None

    def test_plot_balance_assessment(self, sample_dataset):
        """Test balance assessment plotting."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Test plotting
        fig = psm.plot_balance_assessment()
        assert fig is not None

    def test_plot_treatment_effects(self, sample_dataset):
        """Test treatment effects plotting."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)
        psm.estimate_treatment_effects(outcome_cols="filed_2024")

        # Test plotting
        fig = psm.plot_treatment_effects()
        assert fig is not None

    def test_generate_summary_report(self, sample_dataset):
        """Test summary report generation."""
        psm = PropensityScoreMatching(sample_dataset)

        # Test report before analysis
        report = psm.generate_summary_report()
        assert isinstance(report, str)
        assert "No analysis performed" in report

        # Test report after complete analysis
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)
        psm.assess_balance()
        psm.estimate_treatment_effects(outcome_cols="filed_2024")

        report = psm.generate_summary_report()
        assert isinstance(report, str)
        assert "PROPENSITY SCORE MATCHING ANALYSIS SUMMARY" in report
        assert "PROPENSITY SCORE MODEL" in report
        assert "MATCHING RESULTS" in report


class TestMatchingMethods:
    """Test different matching algorithms."""

    def test_nearest_neighbor_matching_with_replacement(self, sample_dataset):
        """Test nearest neighbor matching with replacement."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()

        results = psm.perform_matching(
            method="nearest_neighbor",
            replacement=True,
            ratio=2,  # 1:2 matching
        )

        assert results["method"] == "nearest_neighbor"
        # With replacement, should be able to match more units
        assert results["n_control_matched"] >= results["n_treated_matched"]

    def test_matching_with_small_caliper(self, sample_dataset):
        """Test matching with very strict caliper."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()

        # Very strict caliper
        results = psm.perform_matching(
            method="nearest_neighbor",
            caliper=0.01,  # Very small
        )

        # Should result in lower matching rate
        assert 0 <= results["matching_rate"] <= 1

    def test_invalid_matching_method(self, sample_dataset):
        """Test error handling for invalid matching method."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()

        with pytest.raises(ValueError, match="Unknown matching method"):
            psm.perform_matching(method="invalid_method")


class TestDataHandling:
    """Test data handling and edge cases."""

    def test_categorical_variables_handling(self, sample_dataset):
        """Test handling of categorical variables in propensity score estimation."""
        psm = PropensityScoreMatching(sample_dataset)

        # Include categorical variables
        covariates = [
            "age",
            "tech_savviness",
            "income_bracket",
            "device_type",
            "user_type",
        ]
        available_covariates = [
            col for col in covariates if col in sample_dataset.columns
        ]

        if len(available_covariates) > 2:
            results = psm.estimate_propensity_scores(covariates=available_covariates)
            assert psm.propensity_scores is not None

    def test_missing_covariates(self, sample_dataset):
        """Test handling when specified covariates don't exist."""
        psm = PropensityScoreMatching(sample_dataset)

        # Try with non-existent covariates
        with pytest.raises(ValueError, match="No valid covariates found"):
            psm.estimate_propensity_scores(covariates=["nonexistent_var"])

    def test_empty_treatment_or_control_groups(self):
        """Test behavior with edge case data."""
        # Create data with only treated units
        data = pd.DataFrame(
            {
                "used_smart_assistant": [1] * 50,  # All treated
                "filed_2024": np.random.binomial(1, 0.7, 50),
                "age": np.random.normal(35, 10, 50),
                "tech_savviness": np.random.normal(50, 15, 50),
            }
        )

        psm = PropensityScoreMatching(data)

        # Should handle gracefully (will raise error for single class)
        with pytest.raises(ValueError, match="samples of at least 2 classes"):
            psm.estimate_propensity_scores()

        # Test with mixed data that should work
        mixed_data = pd.DataFrame(
            {
                "used_smart_assistant": [0] * 25 + [1] * 25,  # Mixed treatment
                "filed_2024": np.random.binomial(1, 0.7, 50),
                "age": np.random.normal(35, 10, 50),
                "tech_savviness": np.random.normal(50, 15, 50),
            }
        )

        psm_mixed = PropensityScoreMatching(mixed_data)
        results = psm_mixed.estimate_propensity_scores()
        assert psm_mixed.propensity_scores is not None

        # Matching should handle gracefully
        matching_results = psm_mixed.perform_matching(method="nearest_neighbor")
        assert isinstance(matching_results, dict)

    def test_small_sample_size(self):
        """Test with very small sample size."""
        # Create minimal dataset
        data = pd.DataFrame(
            {
                "used_smart_assistant": [0, 1, 0, 1],
                "filed_2024": [0, 1, 1, 1],
                "age": [25, 35, 45, 55],
                "tech_savviness": [30, 70, 40, 80],
            }
        )

        psm = PropensityScoreMatching(data)
        results = psm.estimate_propensity_scores()

        assert psm.propensity_scores is not None
        assert len(psm.propensity_scores) == 4


class TestBalanceAssessment:
    """Test covariate balance assessment functionality."""

    def test_balance_with_perfect_matching(self):
        """Test balance assessment with artificially perfect matching."""
        # Create data where treatment and control are identical after "matching"
        np.random.seed(42)
        n = 100

        # Create identical units for treatment and control
        data = pd.DataFrame(
            {
                "used_smart_assistant": [0] * (n // 2) + [1] * (n // 2),
                "filed_2024": [0, 1] * (n // 2),
                "age": list(range(25, 25 + n // 2)) * 2,
                "tech_savviness": list(range(30, 30 + n // 2)) * 2,
            }
        )

        psm = PropensityScoreMatching(data)
        psm.estimate_propensity_scores()

        # Manual "matching" - use all data
        psm.matched_data = data.copy()

        balance_results = psm.assess_balance(covariates=["age", "tech_savviness"])

        # After perfect matching, balance should be very good
        after_balance = balance_results["after_matching"]
        for covar in ["age", "tech_savviness"]:
            if covar in after_balance:
                smd = abs(after_balance[covar]["standardized_mean_diff"])
                assert smd < 0.1  # Should be well balanced

    def test_balance_categorical_variables(self, sample_dataset):
        """Test balance assessment with categorical variables."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Include categorical variables
        categorical_vars = [
            col
            for col in ["income_bracket", "device_type", "user_type", "region"]
            if col in sample_dataset.columns
        ]

        if categorical_vars:
            balance_results = psm.assess_balance(
                covariates=categorical_vars[:2]  # Test first two
            )

            # Should handle categorical variables without errors
            assert "before_matching" in balance_results
            assert "after_matching" in balance_results


class TestTreatmentEffectEstimation:
    """Test treatment effect estimation methods."""

    def test_simple_difference_estimator(self, sample_dataset):
        """Test simple difference estimator details."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Estimate effects
        effects = psm.estimate_treatment_effects(
            outcome_cols="filed_2024", method="simple_difference"
        )

        result = effects["filed_2024"]

        # Check that means are calculated
        assert "treated_mean" in result
        assert "control_mean" in result
        assert "standard_error" in result

        # ATE should equal difference in means
        expected_ate = result["treated_mean"] - result["control_mean"]
        assert abs(result["ate"] - expected_ate) < 1e-10

    def test_treatment_effects_with_missing_outcomes(self, sample_dataset):
        """Test handling of missing outcome values."""
        # Create data with some missing outcomes
        data_with_missing = sample_dataset.copy()
        if "satisfaction_2024" in data_with_missing.columns:
            # Introduce some missing values
            missing_idx = np.random.choice(
                data_with_missing.index,
                size=len(data_with_missing) // 10,
                replace=False,
            )
            data_with_missing.loc[missing_idx, "satisfaction_2024"] = np.nan

            psm = PropensityScoreMatching(data_with_missing)
            psm.estimate_propensity_scores()
            psm.perform_matching(method="nearest_neighbor", caliper=0.2)

            # Should handle missing values gracefully
            effects = psm.estimate_treatment_effects(outcome_cols="satisfaction_2024")

            result = effects["satisfaction_2024"]
            # Sample sizes should be less than total due to missing values
            assert result["n_treated"] + result["n_control"] <= len(psm.matched_data)

    def test_invalid_estimation_method(self, sample_dataset):
        """Test error handling for invalid estimation method."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        with pytest.raises(ValueError, match="Unknown estimation method"):
            psm.estimate_treatment_effects(
                outcome_cols="filed_2024", method="invalid_method"
            )


class TestConvenienceFunction:
    """Test the convenience function for complete PSM analysis."""

    def test_load_and_analyze_psm(self, real_config_path, tmp_path):
        """Test complete PSM analysis convenience function."""
        # Create temporary CSV file
        from src.data_simulation import TaxSoftwareDataSimulator

        simulator = TaxSoftwareDataSimulator(n_users=100, config_path=real_config_path)
        df = simulator.generate_complete_dataset()

        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        # Test convenience function
        psm = load_and_analyze_psm(
            file_path=str(csv_path),
            treatment_col="used_smart_assistant",
            outcome_cols="filed_2024",
            matching_method="nearest_neighbor",
            caliper=0.1,
        )

        # Should return fitted PSM object
        assert isinstance(psm, PropensityScoreMatching)
        assert psm.propensity_scores is not None
        assert psm.matched_data is not None
        assert psm.treatment_effects is not None

    def test_load_and_analyze_psm_multiple_outcomes(self, real_config_path, tmp_path):
        """Test convenience function with multiple outcomes."""
        # Create temporary CSV file
        from src.data_simulation import TaxSoftwareDataSimulator

        simulator = TaxSoftwareDataSimulator(n_users=150, config_path=real_config_path)
        df = simulator.generate_complete_dataset()

        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        # Test with multiple outcomes
        available_outcomes = [
            col
            for col in ["filed_2024", "satisfaction_2024", "time_to_complete_2024"]
            if col in df.columns
        ]

        if len(available_outcomes) > 1:
            psm = load_and_analyze_psm(
                file_path=str(csv_path),
                outcome_cols=available_outcomes[:2],  # Test first two
                matching_method="caliper",
                caliper=0.15,
            )

            assert len(psm.treatment_effects) >= 1


class TestRobustnessAndEdgeCases:
    """Test robustness and edge cases."""

    def test_perfect_separation(self):
        """Test handling of perfect separation in propensity score model."""
        # Create data with perfect separation
        n = 100
        data = pd.DataFrame(
            {
                "used_smart_assistant": [0] * (n // 2) + [1] * (n // 2),
                "filed_2024": np.random.binomial(1, 0.5, n),
                "perfect_predictor": [0] * (n // 2)
                + [1] * (n // 2),  # Perfect separation
                "age": np.random.normal(35, 10, n),
            }
        )

        psm = PropensityScoreMatching(data)

        # Should handle gracefully (may issue warnings)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = psm.estimate_propensity_scores(
                covariates=["perfect_predictor", "age"]
            )

        assert psm.propensity_scores is not None

    def test_all_same_propensity_score(self):
        """Test handling when all units have same propensity score."""
        # Create data where propensity scores will be very similar
        n = 50
        data = pd.DataFrame(
            {
                "used_smart_assistant": np.random.binomial(1, 0.5, n),
                "filed_2024": np.random.binomial(1, 0.5, n),
                "constant_var": [1] * n,  # No variation
                "age": np.random.normal(35, 0.1, n),  # Very little variation
            }
        )

        psm = PropensityScoreMatching(data)
        results = psm.estimate_propensity_scores(covariates=["constant_var", "age"])

        # Should still work
        assert psm.propensity_scores is not None

        # Matching might have issues but shouldn't crash
        matching_results = psm.perform_matching(method="nearest_neighbor")
        assert isinstance(matching_results, dict)

    def test_numerical_precision(self, sample_dataset):
        """Test numerical precision in calculations."""
        psm = PropensityScoreMatching(sample_dataset)
        psm.estimate_propensity_scores()

        # Test that propensity scores are in valid range
        assert np.all(psm.propensity_scores >= 0)
        assert np.all(psm.propensity_scores <= 1)

        # Test that no NaN values are produced
        assert not np.any(np.isnan(psm.propensity_scores))

    def test_reproducibility(self, sample_dataset):
        """Test that results are reproducible with same random seed."""
        # First run
        psm1 = PropensityScoreMatching(sample_dataset)
        results1 = psm1.estimate_propensity_scores(model_params={"random_state": 42})

        # Second run with same seed
        psm2 = PropensityScoreMatching(sample_dataset)
        results2 = psm2.estimate_propensity_scores(model_params={"random_state": 42})

        # Should produce identical results
        np.testing.assert_array_equal(psm1.propensity_scores, psm2.propensity_scores)
        assert results1["auc_score"] == results2["auc_score"]
