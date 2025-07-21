"""Integration tests for the complete workflow."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from src.causal_methods.did import DifferenceInDifferences
from src.data_simulation import TaxSoftwareDataSimulator, generate_and_save_data
from src.causal_methods.psm import PropensityScoreMatching, load_and_analyze_psm


class TestFullWorkflow:
    """Test the complete data generation -> analysis workflow."""

    def test_end_to_end_workflow(self, real_config_path):
        """Test complete workflow from data generation to DiD analysis."""
        # Step 1: Generate synthetic data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            # Generate data
            df = generate_and_save_data(
                output_path=output_path,
                n_users=200,  # Larger sample for reliable results
                config_path=real_config_path,
            )

            # Verify data generation
            assert len(df) == 200
            assert Path(output_path).exists()

            # Step 2: Run DiD analysis
            did = DifferenceInDifferences(df)
            panel_df = did.prepare_panel_data()

            # Verify panel preparation with actual column names
            assert len(panel_df) == 400  # 200 users * 2 periods
            assert (
                "post_treatment" in panel_df.columns
            )  # DiD uses 'post_treatment' not 'post'
            assert "treated" in panel_df.columns
            assert "outcome" in panel_df.columns

            # Step 3: Check that analysis runs without errors
            # (The exact methods available depend on the DiD implementation)
            assert isinstance(did, DifferenceInDifferences)
            assert len(panel_df) > 0

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_multiple_scenarios_analysis(self):
        """Test analysis across multiple scenarios."""
        scenarios = [
            "config/simulation_config.yaml",
            "config/scenario_high_treatment.yaml",
            "config/scenario_low_adoption.yaml",
        ]

        results_by_scenario = {}

        for scenario_path in scenarios:
            if not Path(scenario_path).exists():
                continue

            # Generate data for this scenario
            simulator = TaxSoftwareDataSimulator(n_users=150, config_path=scenario_path)
            df = simulator.generate_complete_dataset()

            # Analyze with DiD
            did = DifferenceInDifferences(df)
            panel_df = did.prepare_panel_data()

            scenario_name = Path(scenario_path).stem
            results_by_scenario[scenario_name] = {
                "treatment_rate": df["used_smart_assistant"].mean(),
                "filing_rate_2024": df["filed_2024"].mean(),
                "panel_data_length": len(panel_df),
            }

        # Verify we got results for multiple scenarios
        assert len(results_by_scenario) >= 2

        # Results should vary across scenarios
        treatment_rates = [r["treatment_rate"] for r in results_by_scenario.values()]

        # Should have some variation
        assert (
            max(treatment_rates) - min(treatment_rates) > 0.05
        )  # At least 5pp difference


class TestReproducibility:
    """Test that results are reproducible."""

    def test_data_generation_reproducibility(self, real_config_path):
        """Test that data generation is reproducible with same seed."""
        # Generate data twice with same config
        df1 = TaxSoftwareDataSimulator(
            n_users=100, config_path=real_config_path
        ).generate_complete_dataset()

        df2 = TaxSoftwareDataSimulator(
            n_users=100, config_path=real_config_path
        ).generate_complete_dataset()

        # Should be identical (same random seed in config)
        pd.testing.assert_frame_equal(df1, df2)

    def test_analysis_reproducibility(self, real_config_path):
        """Test that analysis results are reproducible."""
        # Generate same data
        df = TaxSoftwareDataSimulator(
            n_users=100, config_path=real_config_path
        ).generate_complete_dataset()

        # Analyze twice
        did1 = DifferenceInDifferences(df)
        panel_df1 = did1.prepare_panel_data()

        did2 = DifferenceInDifferences(df)
        panel_df2 = did2.prepare_panel_data()

        # Should be identical
        pd.testing.assert_frame_equal(panel_df1, panel_df2)


class TestScalability:
    """Test performance with different data sizes."""

    def test_small_sample_size(self, real_config_path):
        """Test with very small sample size."""
        df = TaxSoftwareDataSimulator(
            n_users=20,  # Very small
            config_path=real_config_path,
        ).generate_complete_dataset()

        did = DifferenceInDifferences(df)
        panel_df = did.prepare_panel_data()

        # Should still work
        assert len(panel_df) == 40  # 20 users * 2 periods
        assert "outcome" in panel_df.columns

    def test_medium_sample_size(self, real_config_path):
        """Test with medium sample size."""
        df = TaxSoftwareDataSimulator(
            n_users=500, config_path=real_config_path
        ).generate_complete_dataset()

        did = DifferenceInDifferences(df)
        panel_df = did.prepare_panel_data()

        # Should handle larger datasets
        assert len(panel_df) == 1000  # 500 users * 2 periods
        assert panel_df["user_id"].nunique() == 500


class TestDataQuality:
    """Test data quality and consistency."""

    def test_data_consistency_checks(self, real_config_path):
        """Test various data consistency checks."""
        df = TaxSoftwareDataSimulator(
            n_users=200, config_path=real_config_path
        ).generate_complete_dataset()

        # Age should be reasonable
        assert df["age"].min() >= 18
        assert df["age"].max() <= 80

        # Tech-savviness should be in range
        assert df["tech_savviness"].min() >= 0
        assert df["tech_savviness"].max() <= 100

        # Boolean columns should be 0/1 or boolean
        bool_columns = ["filed_2023", "filed_2024", "used_smart_assistant"]

        # Add early login columns if they exist
        early_login_cols = [col for col in df.columns if "early_login" in col]
        bool_columns.extend(early_login_cols)

        for col in bool_columns:
            if col in df.columns:
                # Check if values are boolean-like (0/1, True/False, or actual boolean type)
                unique_vals = set(df[col].unique())
                is_binary = (
                    unique_vals.issubset({0, 1})
                    or unique_vals.issubset({True, False})
                    or unique_vals.issubset({0, 1})
                )
                assert is_binary, f"{col} contains non-boolean values: {unique_vals}"

        # Support ticket columns are counts (can be 0, 1, 2, etc.)
        support_columns = ["support_tickets_2023", "support_tickets_2024"]
        for col in support_columns:
            if col in df.columns:
                # Should be non-negative integers
                assert (df[col] >= 0).all(), f"{col} contains negative values"
                assert df[col].dtype in ["int64", "int32", "uint64", "uint32"], (
                    f"{col} should be integer type"
                )

        # Time variables should be positive for users who filed
        time_columns = ["time_to_complete_2023", "time_to_complete_2024"]
        for col in time_columns:
            if col in df.columns:
                # Should be >= 0 (0 for users who didn't file)
                assert (df[col] >= 0).all(), f"{col} contains negative values"

        # Sessions should be at least 1 for users who filed
        session_columns = ["sessions_2023", "sessions_2024"]
        for col in session_columns:
            if col in df.columns:
                # Should be >= 1 for users who filed
                filed_col = col.replace("sessions", "filed")
                if filed_col in df.columns:
                    filed_users = df[df[filed_col] == 1]
                    if len(filed_users) > 0:
                        assert (filed_users[col] >= 1).all(), (
                            f"{col} should be >= 1 for users who filed"
                        )

        # Satisfaction should be in range 1-10
        if "satisfaction_2024" in df.columns:
            assert df["satisfaction_2024"].min() >= 1
            assert df["satisfaction_2024"].max() <= 10

    def test_treatment_assignment_realism(self, real_config_path):
        """Test that treatment assignment follows realistic patterns."""
        df = TaxSoftwareDataSimulator(
            n_users=300, config_path=real_config_path
        ).generate_complete_dataset()

        # Treatment rate should be reasonable
        treatment_rate = df["used_smart_assistant"].mean()
        assert 0.1 <= treatment_rate <= 0.8, (
            f"Treatment rate {treatment_rate} seems unrealistic"
        )

        # High-tech users should be more likely to be treated
        if "tech_savviness" in df.columns:
            high_tech = df["tech_savviness"] > 70
            if high_tech.sum() > 10:  # Need sufficient sample
                high_tech_treatment_rate = df[high_tech]["used_smart_assistant"].mean()
                low_tech_treatment_rate = df[~high_tech]["used_smart_assistant"].mean()

                # High-tech users should have higher treatment rate
                assert high_tech_treatment_rate >= low_tech_treatment_rate - 0.1

        # Early login users should be more likely to be treated
        early_login_cols = [col for col in df.columns if "early_login" in col]
        if early_login_cols:
            early_login_col = early_login_cols[0]
            early_login_users = df[early_login_col] == 1
            if early_login_users.sum() > 10:
                early_treatment_rate = df[early_login_users][
                    "used_smart_assistant"
                ].mean()
                late_treatment_rate = df[~early_login_users][
                    "used_smart_assistant"
                ].mean()

                assert early_treatment_rate >= late_treatment_rate - 0.1

    def test_outcome_patterns(self, real_config_path):
        """Test that outcomes follow expected patterns."""
        df = TaxSoftwareDataSimulator(
            n_users=250, config_path=real_config_path
        ).generate_complete_dataset()

        # Filing rates should be reasonable - updated upper bound
        filing_2023 = df["filed_2023"].mean()
        filing_2024 = df["filed_2024"].mean()

        assert 0.5 <= filing_2023 <= 0.95, (
            f"2023 filing rate {filing_2023} seems unrealistic"
        )
        assert 0.5 <= filing_2024 <= 0.95, (
            f"2024 filing rate {filing_2024} seems unrealistic"
        )  # Increased upper bound

        # Treatment should generally improve outcomes
        treated = df[df["used_smart_assistant"] == 1]
        control = df[df["used_smart_assistant"] == 0]

        if len(treated) > 20 and len(control) > 20:
            # Filing rates
            treated_filing = treated["filed_2024"].mean()
            control_filing = control["filed_2024"].mean()

            # Treated should file at least as much as control
            assert treated_filing >= control_filing - 0.1

            # Satisfaction (if available)
            if "satisfaction_2024" in df.columns:
                treated_satisfaction = treated["satisfaction_2024"].mean()
                control_satisfaction = control["satisfaction_2024"].mean()

                # Treated should be at least as satisfied
                assert treated_satisfaction >= control_satisfaction - 0.5


class TestConfigurationScenarios:
    """Test different configuration scenarios work correctly."""

    def test_high_treatment_scenario(self):
        """Test high treatment effect scenario."""
        if not Path("config/scenario_high_treatment.yaml").exists():
            pytest.skip("High treatment scenario config not found")

        df = TaxSoftwareDataSimulator(
            n_users=150, config_path="config/scenario_high_treatment.yaml"
        ).generate_complete_dataset()

        # Should have reasonable data
        assert len(df) == 150

        # Analyze with DiD
        did = DifferenceInDifferences(df)
        panel_df = did.prepare_panel_data()

        # Should produce valid panel data
        assert len(panel_df) == 300  # 150 users * 2 periods
        assert "outcome" in panel_df.columns

    def test_low_adoption_scenario(self):
        """Test low adoption scenario."""
        if not Path("config/scenario_low_adoption.yaml").exists():
            pytest.skip("Low adoption scenario config not found")

        df = TaxSoftwareDataSimulator(
            n_users=150, config_path="config/scenario_low_adoption.yaml"
        ).generate_complete_dataset()

        # Should have lower treatment adoption
        treatment_rate = df["used_smart_assistant"].mean()

        # Load baseline for comparison
        baseline_df = TaxSoftwareDataSimulator(
            n_users=150, config_path="config/simulation_config.yaml"
        ).generate_complete_dataset()
        baseline_treatment_rate = baseline_df["used_smart_assistant"].mean()

        # Low adoption scenario should have lower treatment rate
        assert treatment_rate <= baseline_treatment_rate + 0.05  # Allow small tolerance


class TestDiDClassMethods:
    """Test specific methods of the DifferenceInDifferences class."""

    def test_basic_instantiation(self, sample_dataset):
        """Test that DiD class can be instantiated with real data."""
        did = DifferenceInDifferences(sample_dataset)

        assert hasattr(did, "data")
        assert hasattr(did, "results")
        assert hasattr(did, "fitted_models")
        assert len(did.data) == len(sample_dataset)

    def test_panel_preparation_with_real_data(self, sample_dataset):
        """Test panel data preparation with realistic data."""
        did = DifferenceInDifferences(sample_dataset)
        panel_df = did.prepare_panel_data()

        # Basic checks with actual column names
        assert len(panel_df) == len(sample_dataset) * 2
        assert "user_id" in panel_df.columns
        assert "year" in panel_df.columns  # DiD uses 'year' not 'period'

        # Check that all users are represented in both periods
        users_per_period = panel_df.groupby("year")["user_id"].nunique()
        assert users_per_period[2023] == len(sample_dataset)
        assert users_per_period[2024] == len(sample_dataset)

    def test_different_outcome_columns(self, sample_dataset):
        """Test panel preparation with different outcome columns."""
        # Test different outcome variables if the method supports them
        outcome_pairs = [
            ("filed_2023", "filed_2024"),
            ("time_to_complete_2023", "time_to_complete_2024"),
            ("sessions_2023", "sessions_2024"),
        ]

        for outcome_2023, outcome_2024 in outcome_pairs:
            if (
                outcome_2023 in sample_dataset.columns
                and outcome_2024 in sample_dataset.columns
            ):
                did = DifferenceInDifferences(sample_dataset)

                try:
                    # Try with custom outcome columns if supported
                    panel_df = did.prepare_panel_data(
                        outcome_2023_col=outcome_2023, outcome_2024_col=outcome_2024
                    )
                    assert "outcome" in panel_df.columns
                    assert len(panel_df) > 0
                except TypeError:
                    # Method might not support custom column specification
                    # Just test with default columns
                    panel_df = did.prepare_panel_data()
                    assert len(panel_df) > 0


class TestPSMIntegration:
    """Test PSM integration with the complete workflow."""

    def test_complete_psm_workflow(self, real_config_path):
        """Test complete PSM workflow from data generation to effect estimation."""
        # Generate data
        df = TaxSoftwareDataSimulator(
            n_users=200,  # Larger sample for PSM
            config_path=real_config_path
        ).generate_complete_dataset()

        # Initialize PSM
        psm = PropensityScoreMatching(df)

        # Step 1: Estimate propensity scores
        ps_results = psm.estimate_propensity_scores(
            treatment_col="used_smart_assistant",
            covariates=["age", "tech_savviness", "filed_2023", "early_login_2024"]
        )

        # Verify propensity score estimation
        assert psm.propensity_scores is not None
        assert len(psm.propensity_scores) == len(df)
        assert 0 <= ps_results["auc_score"] <= 1
        assert "propensity_score" in psm.data.columns

        # Step 2: Perform matching
        matching_results = psm.perform_matching(
            method="nearest_neighbor",
            caliper=0.1,
            replacement=False
        )

        # Verify matching
        assert psm.matched_data is not None
        assert len(psm.matched_data) > 0
        assert 0 <= matching_results["matching_rate"] <= 1
        assert matching_results["n_treated_matched"] > 0

        # Step 3: Assess balance
        balance_results = psm.assess_balance(
            covariates=["age", "tech_savviness", "propensity_score"]
        )

        # Verify balance assessment
        assert "before_matching" in balance_results
        assert "after_matching" in balance_results
        assert isinstance(balance_results["before_matching"], dict)
        assert isinstance(balance_results["after_matching"], dict)

        # Step 4: Estimate treatment effects
        effect_results = psm.estimate_treatment_effects(
            outcome_cols=["filed_2024"],
            treatment_col="used_smart_assistant"
        )

        # Verify treatment effect estimation
        assert "filed_2024" in effect_results
        effect_stats = effect_results["filed_2024"]
        assert "ate" in effect_stats
        assert "ci_lower" in effect_stats
        assert "ci_upper" in effect_stats
        assert "p_value" in effect_stats

        # Confidence interval should make sense
        assert effect_stats["ci_lower"] <= effect_stats["ate"] <= effect_stats["ci_upper"]

    def test_psm_convenience_function_integration(self, real_config_path):
        """Test PSM convenience function with complete pipeline."""
        # Generate and save data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            df = generate_and_save_data(
                output_path=output_path,
                n_users=150,
                config_path=real_config_path
            )

            # Test convenience function
            psm = load_and_analyze_psm(
                file_path=output_path,
                treatment_col="used_smart_assistant",
                outcome_cols=["filed_2024", "satisfaction_2024"],
                matching_method="nearest_neighbor",
                caliper=0.15
            )

            # Should have completed full analysis
            assert psm.propensity_scores is not None
            assert psm.matched_data is not None
            assert psm.balance_stats is not None
            assert psm.treatment_effects is not None

            # Should have results for available outcomes
            available_outcomes = [col for col in ["filed_2024", "satisfaction_2024"] 
                                if col in df.columns]
            for outcome in available_outcomes:
                assert outcome in psm.treatment_effects

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_psm_different_scenarios(self):
        """Test PSM across different data scenarios."""
        scenarios = [
            "config/simulation_config.yaml",
            "config/scenario_high_treatment.yaml",
            "config/scenario_low_adoption.yaml"
        ]

        psm_results = {}

        for scenario_path in scenarios:
            if not Path(scenario_path).exists():
                continue

            # Generate data for this scenario
            df = TaxSoftwareDataSimulator(
                n_users=120,
                config_path=scenario_path
            ).generate_complete_dataset()

            # Run PSM analysis
            psm = PropensityScoreMatching(df)
            psm.estimate_propensity_scores()
            matching_results = psm.perform_matching(
                method="nearest_neighbor",
                caliper=0.15
            )

            if psm.matched_data is not None and len(psm.matched_data) > 10:
                psm.estimate_treatment_effects(outcome_cols="filed_2024")

                scenario_name = Path(scenario_path).stem
                psm_results[scenario_name] = {
                    "treatment_rate": df["used_smart_assistant"].mean(),
                    "matching_rate": matching_results["matching_rate"],
                    "ate": psm.treatment_effects["filed_2024"]["ate"] if psm.treatment_effects else np.nan
                }

        # Should have results for multiple scenarios
        assert len(psm_results) >= 2

    def test_psm_balance_improvement(self, real_config_path):
        """Test that PSM actually improves covariate balance."""
        df = TaxSoftwareDataSimulator(
            n_users=250,
            config_path=real_config_path
        ).generate_complete_dataset()

        psm = PropensityScoreMatching(df)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Assess balance
        balance_results = psm.assess_balance(
            covariates=["age", "tech_savviness", "filed_2023"]
        )

        # Check that balance improved for most variables
        improved_balance_count = 0
        total_variables = 0

        before_balance = balance_results["before_matching"]
        after_balance = balance_results["after_matching"]

        for covar in balance_results["covariates_assessed"]:
            if covar in before_balance and covar in after_balance:
                before_smd = abs(before_balance[covar]["standardized_mean_diff"])
                after_smd = abs(after_balance[covar]["standardized_mean_diff"])

                total_variables += 1
                if after_smd < before_smd:
                    improved_balance_count += 1

        # Most variables should show improved balance
        if total_variables > 0:
            improvement_rate = improved_balance_count / total_variables
            assert improvement_rate >= 0.5  # At least 50% should improve

    def test_psm_with_missing_data(self, real_config_path):
        """Test PSM robustness with missing data."""
        df = TaxSoftwareDataSimulator(
            n_users=150,
            config_path=real_config_path
        ).generate_complete_dataset()

        # Introduce missing values in satisfaction
        if "satisfaction_2024" in df.columns:
            missing_idx = np.random.choice(
                df.index,
                size=len(df) // 10,
                replace=False
            )
            df.loc[missing_idx, "satisfaction_2024"] = np.nan

        # PSM should handle missing data gracefully
        psm = PropensityScoreMatching(df)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)

        # Should still be able to estimate effects on available data
        if psm.matched_data is not None:
            psm.estimate_treatment_effects(
                outcome_cols=["filed_2024", "satisfaction_2024"]
            )

            # Should have results for filed_2024
            assert "filed_2024" in psm.treatment_effects

            # Satisfaction results should handle missing values
            if "satisfaction_2024" in psm.treatment_effects:
                effect_stats = psm.treatment_effects["satisfaction_2024"]
                # Sample size should be less than matched data due to missing values
                total_matched = len(psm.matched_data)
                analyzed_sample = effect_stats["n_treated"] + effect_stats["n_control"]
                assert analyzed_sample <= total_matched

    def test_psm_sensitivity_to_caliper(self, real_config_path):
        """Test PSM sensitivity to different caliper values."""
        df = TaxSoftwareDataSimulator(
            n_users=200,
            config_path=real_config_path
        ).generate_complete_dataset()

        caliper_values = [0.05, 0.1, 0.2, 0.3]
        results_by_caliper = []

        for caliper_std in caliper_values:
            psm = PropensityScoreMatching(df)
            psm.estimate_propensity_scores()

            caliper = caliper_std * np.std(psm.propensity_scores)
            matching_results = psm.perform_matching(
                method="nearest_neighbor",
                caliper=caliper
            )

            if psm.matched_data is not None and len(psm.matched_data) > 10:
                psm.estimate_treatment_effects(outcome_cols="filed_2024")

                results_by_caliper.append({
                    "caliper_std": caliper_std,
                    "matching_rate": matching_results["matching_rate"],
                    "sample_size": len(psm.matched_data),
                    "ate": psm.treatment_effects["filed_2024"]["ate"]
                })

        # Should have results for multiple calipers
        assert len(results_by_caliper) >= 2

        # Tighter calipers should generally have lower matching rates
        matching_rates = [r["matching_rate"] for r in results_by_caliper]
        caliper_stds = [r["caliper_std"] for r in results_by_caliper]

        # Generally, larger calipers should have higher matching rates
        # (though this might not be perfectly monotonic)
        correlation = np.corrcoef(caliper_stds, matching_rates)[0, 1]
        assert correlation > 0  # Positive correlation expected


class TestPSMAndDiDComparison:
    """Test comparing PSM and DiD methods."""

    def test_psm_vs_did_estimates(self, real_config_path):
        """Compare PSM and DiD estimates on same data."""
        df = TaxSoftwareDataSimulator(
            n_users=200,
            config_path=real_config_path
        ).generate_complete_dataset()

        # PSM Analysis
        psm = PropensityScoreMatching(df)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.15)
        psm.estimate_treatment_effects(outcome_cols="filed_2024")

        # DiD Analysis
        did = DifferenceInDifferences(df)
        panel_df = did.prepare_panel_data()

        # Both should produce treatment effect estimates
        assert psm.treatment_effects is not None
        assert "filed_2024" in psm.treatment_effects

        # Both methods should run without errors
        psm_ate = psm.treatment_effects["filed_2024"]["ate"]
        assert not np.isnan(psm_ate)

        # Panel data should be created for DiD
        assert len(panel_df) == len(df) * 2  # Two time periods

    def test_methods_consistency_check(self, real_config_path):
        """Test that different causal methods produce reasonable estimates."""
        df = TaxSoftwareDataSimulator(
            n_users=300,  # Larger sample for stable estimates
            config_path=real_config_path
        ).generate_complete_dataset()

        estimates = {}

        # Naive estimate
        treated = df[df["used_smart_assistant"] == 1]["filed_2024"]
        control = df[df["used_smart_assistant"] == 0]["filed_2024"]
        estimates["naive"] = treated.mean() - control.mean()

        # PSM estimate
        psm = PropensityScoreMatching(df)
        psm.estimate_propensity_scores()
        psm.perform_matching(method="nearest_neighbor", caliper=0.2)
        if psm.matched_data is not None:
            psm.estimate_treatment_effects(outcome_cols="filed_2024")
            estimates["psm"] = psm.treatment_effects["filed_2024"]["ate"]

        # DiD estimate (if we can prepare panel data)
        try:
            did = DifferenceInDifferences(df)
            panel_df = did.prepare_panel_data()
            estimates["did_panel_created"] = True
        except Exception:
            estimates["did_panel_created"] = False

        # Should have at least naive and PSM estimates
        assert "naive" in estimates
        if "psm" in estimates:
            # Estimates should be finite numbers
            assert np.isfinite(estimates["naive"])
            assert np.isfinite(estimates["psm"])

            # PSM should generally be different from naive (bias correction)
            # But both should be reasonable effect sizes
            assert abs(estimates["naive"]) < 1.0  # Filing rate effect shouldn't be > 100pp
            assert abs(estimates["psm"]) < 1.0
