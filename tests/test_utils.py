"""Tests for utility functions and edge cases."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.data_simulation import load_config, merge_configs


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_load_config_with_inheritance(self):
        """Test config loading with inheritance from scenario files."""
        # Test that scenario configs properly inherit from base
        if Path("config/scenario_high_treatment.yaml").exists():
            config = load_config("config/scenario_high_treatment.yaml")

            # Should have all required sections
            required_sections = [
                "simulation",
                "demographics",
                "tech_savviness",
                "baseline_2023",
                "treatment",
                "outcomes_2024",
            ]

            for section in required_sections:
                assert section in config, f"Missing section: {section}"

            # Should have overridden values
            assert config["simulation"]["random_seed"] == 123

            # Should have inherited values not specified in scenario
            assert "base_score" in config["tech_savviness"]

    def test_config_yaml_syntax(self):
        """Test that all config files have valid YAML syntax."""
        config_files = [
            "config/simulation_config.yaml",
            "config/scenario_high_treatment.yaml",
            "config/scenario_low_adoption.yaml",
        ]

        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file) as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")

    def test_config_required_fields(self):
        """Test that config files have all required fields."""
        if Path("config/simulation_config.yaml").exists():
            config = load_config("config/simulation_config.yaml")

            # Required top-level sections
            required_sections = [
                "simulation",
                "demographics",
                "tech_savviness",
                "baseline_2023",
                "early_login",
                "treatment",
                "outcomes_2024",
                "output",
            ]

            for section in required_sections:
                assert section in config, f"Missing required section: {section}"

            # Required simulation fields
            sim_fields = ["random_seed", "default_n_users"]
            for field in sim_fields:
                assert field in config["simulation"], (
                    f"Missing simulation field: {field}"
                )

    def test_probability_weights_sum_to_one(self):
        """Test that probability weights in config sum to 1.0."""
        if Path("config/simulation_config.yaml").exists():
            config = load_config("config/simulation_config.yaml")

            # Check demographic distributions
            demographic_fields = [
                "income_brackets",
                "device_types",
                "user_types",
                "regions",
            ]

            for field in demographic_fields:
                if field in config["demographics"]:
                    weights = config["demographics"][field]["weights"]
                    weight_sum = sum(weights)
                    assert abs(weight_sum - 1.0) < 1e-6, (
                        f"{field} weights sum to {weight_sum}, not 1.0"
                    )


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling across modules."""

    def test_empty_config_handling(self):
        """Test handling of empty or minimal configs."""
        minimal_config = {
            "simulation": {"random_seed": 42, "default_n_users": 10},
            "demographics": {
                "income_brackets": {"values": ["low"], "weights": [1.0]},
                "device_types": {"values": ["mobile"], "weights": [1.0]},
                "user_types": {"values": ["new"], "weights": [1.0]},
                "regions": {"values": ["region1"], "weights": [1.0]},
                "age": {"mean": 40, "std": 10, "min_age": 20, "max_age": 60},
            },
        }

        # Should not crash when merging with base config
        base_config = (
            load_config("config/simulation_config.yaml")
            if Path("config/simulation_config.yaml").exists()
            else {}
        )
        merged = merge_configs(base_config, minimal_config)

        assert "simulation" in merged
        assert merged["simulation"]["random_seed"] == 42

    def test_extreme_sample_sizes(self):
        """Test behavior with extreme sample sizes."""
        # Very small sample size (edge case)
        if Path("config/simulation_config.yaml").exists():
            from src.data_simulation import TaxSoftwareDataSimulator

            # Should handle n_users=1 without crashing
            simulator = TaxSoftwareDataSimulator(
                n_users=1, config_path="config/simulation_config.yaml"
            )
            df = simulator.generate_complete_dataset()
            assert len(df) == 1

            # Should handle n_users=2 (minimal for comparison)
            simulator = TaxSoftwareDataSimulator(
                n_users=2, config_path="config/simulation_config.yaml"
            )
            df = simulator.generate_complete_dataset()
            assert len(df) == 2

    def test_boundary_value_conditions(self):
        """Test boundary conditions for parameter values."""
        # Test with extreme age ranges
        config = {
            "simulation": {"random_seed": 42, "default_n_users": 10},
            "demographics": {
                "age": {
                    "mean": 18,
                    "std": 1,
                    "min_age": 18,
                    "max_age": 19,
                }  # Very narrow range
            },
        }

        # Should handle without crashing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            # This should handle gracefully
            from src.data_simulation import TaxSoftwareDataSimulator

            base_config = (
                load_config("config/simulation_config.yaml")
                if Path("config/simulation_config.yaml").exists()
                else config
            )
            merged_config = merge_configs(base_config, config)

            # Write merged config back
            with open(temp_path, "w") as f:
                yaml.dump(merged_config, f)

            simulator = TaxSoftwareDataSimulator(n_users=5, config_path=temp_path)
            df = simulator.generate_complete_dataset()

            # Ages should be within specified range
            assert df["age"].min() >= 18
            assert df["age"].max() <= 19

        finally:
            Path(temp_path).unlink()


class TestDataValidation:
    """Test data validation and consistency checks."""

    def test_categorical_data_consistency(self):
        """Test that categorical data matches config specifications."""
        if Path("config/simulation_config.yaml").exists():
            from src.data_simulation import TaxSoftwareDataSimulator

            config = load_config("config/simulation_config.yaml")
            simulator = TaxSoftwareDataSimulator(
                n_users=50, config_path="config/simulation_config.yaml"
            )
            df = simulator.generate_complete_dataset()

            # Check income brackets match config
            expected_income = set(config["demographics"]["income_brackets"]["values"])
            actual_income = set(df["income_bracket"].unique())
            assert actual_income.issubset(expected_income), (
                f"Unexpected income values: {actual_income - expected_income}"
            )

            # Check device types match config
            expected_devices = set(config["demographics"]["device_types"]["values"])
            actual_devices = set(df["device_type"].unique())
            assert actual_devices.issubset(expected_devices), (
                f"Unexpected device values: {actual_devices - expected_devices}"
            )

    def test_numeric_ranges(self):
        """Test that numeric variables stay within expected ranges."""
        if Path("config/simulation_config.yaml").exists():
            from src.data_simulation import TaxSoftwareDataSimulator

            simulator = TaxSoftwareDataSimulator(
                n_users=100, config_path="config/simulation_config.yaml"
            )
            df = simulator.generate_complete_dataset()

            # Age should be within configured range
            config = load_config("config/simulation_config.yaml")
            min_age = config["demographics"]["age"]["min_age"]
            max_age = config["demographics"]["age"]["max_age"]

            assert df["age"].min() >= min_age, (
                f"Age {df['age'].min()} below minimum {min_age}"
            )
            assert df["age"].max() <= max_age, (
                f"Age {df['age'].max()} above maximum {max_age}"
            )

            # Tech-savviness should be 0-100
            tech_min = config["tech_savviness"]["min_score"]
            tech_max = config["tech_savviness"]["max_score"]

            assert df["tech_savviness"].min() >= tech_min
            assert df["tech_savviness"].max() <= tech_max

    def test_correlation_patterns(self):
        """Test that expected correlation patterns exist in data."""
        if Path("config/simulation_config.yaml").exists():
            from src.data_simulation import TaxSoftwareDataSimulator

            # Generate larger sample for reliable correlations
            simulator = TaxSoftwareDataSimulator(
                n_users=300, config_path="config/simulation_config.yaml"
            )
            df = simulator.generate_complete_dataset()

            # Tech-savviness should be positively correlated with treatment
            tech_treatment_corr = df["tech_savviness"].corr(df["used_smart_assistant"])
            assert tech_treatment_corr > 0, (
                f"Tech-savviness and treatment should be positively correlated, got {tech_treatment_corr}"
            )

            # Early login should be positively correlated with treatment
            early_login_cols = [col for col in df.columns if "early_login" in col]
            if (
                early_login_cols and df[early_login_cols[0]].var() > 0
            ):  # Check for variation
                early_treatment_corr = df[early_login_cols[0]].corr(
                    df["used_smart_assistant"]
                )
                assert early_treatment_corr > -0.1, (
                    f"Early login and treatment correlation too negative: {early_treatment_corr}"
                )


class TestConfigurationValidation:
    """Test configuration validation and constraints."""

    def test_config_parameter_types(self):
        """Test that config parameters have correct types."""
        if Path("config/simulation_config.yaml").exists():
            config = load_config("config/simulation_config.yaml")

            # Random seed should be integer
            assert isinstance(config["simulation"]["random_seed"], int)

            # Default users should be integer
            assert isinstance(config["simulation"]["default_n_users"], int)

            # Probability values should be floats between 0 and 1
            prob_fields = [
                ("treatment", "base_adoption_rate"),
                ("baseline_2023", "filing", "base_rate"),
                ("outcomes_2024", "filing", "base_rate"),
                ("early_login", "base_probability"),
            ]

            for field_path in prob_fields:
                value = config
                for key in field_path:
                    if key in value:
                        value = value[key]
                    else:
                        break
                else:
                    assert isinstance(value, int | float), (
                        f"Value at {field_path} should be numeric"
                    )
                    assert 0 <= value <= 1, (
                        f"Probability at {field_path} should be between 0 and 1, got {value}"
                    )

    def test_config_consistency_checks(self):
        """Test internal consistency of configuration parameters."""
        if Path("config/simulation_config.yaml").exists():
            config = load_config("config/simulation_config.yaml")

            # Age thresholds should be consistent
            young_threshold = config["tech_savviness"]["age_adjustments"][
                "young_threshold"
            ]
            old_threshold = config["tech_savviness"]["age_adjustments"]["old_threshold"]
            assert young_threshold < old_threshold, (
                "Young threshold should be less than old threshold"
            )

            # Tech score thresholds should be ordered
            tech_high = config["treatment"]["tech_effects"]["high_threshold"]
            tech_medium = config["treatment"]["tech_effects"]["medium_threshold"]
            tech_low = config["treatment"]["tech_effects"]["low_threshold"]

            assert tech_low < tech_medium < tech_high, (
                "Tech thresholds should be ordered: low < medium < high"
            )

            # Treatment probability bounds should be valid
            min_prob = config["treatment"]["min_probability"]
            max_prob = config["treatment"]["max_probability"]
            base_prob = config["treatment"]["base_adoption_rate"]

            assert 0 <= min_prob <= base_prob <= max_prob <= 1, (
                "Treatment probabilities should be ordered: 0 <= min <= base <= max <= 1"
            )
