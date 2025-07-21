"""Tests for data simulation module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_simulation import (
    TaxSoftwareDataSimulator,
    generate_and_save_data,
    load_config,
    merge_configs,
)


class TestConfigLoading:
    """Test configuration loading and merging functions."""

    def test_load_config_with_valid_file(self, temp_config_file):
        """Test loading a valid config file."""
        config = load_config(temp_config_file)

        assert isinstance(config, dict)
        assert "simulation" in config
        assert "demographics" in config
        assert config["simulation"]["random_seed"] == 42

    def test_load_config_with_nonexistent_file(self):
        """Test loading a non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_merge_configs_simple(self):
        """Test simple config merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}

        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 4  # overridden
        assert result["b"]["d"] == 3  # preserved
        assert result["e"] == 5  # added

    def test_merge_configs_deep_nesting(self):
        """Test merging with deep nesting."""
        base = {"level1": {"level2": {"level3": {"value": 1}}}}
        override = {"level1": {"level2": {"level3": {"value": 2, "new": 3}}}}

        result = merge_configs(base, override)

        assert result["level1"]["level2"]["level3"]["value"] == 2
        assert result["level1"]["level2"]["level3"]["new"] == 3

    def test_merge_configs_with_none_values(self):
        """Test merging handles None values correctly."""
        base = {"a": 1, "b": None}
        override = {"a": None, "c": 2}

        result = merge_configs(base, override)

        assert result["a"] is None
        assert result["b"] is None
        assert result["c"] == 2


class TestTaxSoftwareDataSimulator:
    """Test the main data simulator class."""

    def test_init_with_valid_config(self, temp_config_file):
        """Test simulator initialization with valid config."""
        simulator = TaxSoftwareDataSimulator(n_users=100, config_path=temp_config_file)

        assert simulator.n_users == 100
        assert isinstance(simulator.config, dict)
        assert simulator.config["simulation"]["random_seed"] == 42

    def test_init_sets_random_seed(self, temp_config_file):
        """Test that random seed is set correctly."""
        simulator = TaxSoftwareDataSimulator(n_users=50, config_path=temp_config_file)

        # Generate some random data
        data1 = simulator.generate_user_demographics()

        # Create new simulator with same seed
        simulator2 = TaxSoftwareDataSimulator(n_users=50, config_path=temp_config_file)
        data2 = simulator2.generate_user_demographics()

        # Should be identical due to seed
        pd.testing.assert_frame_equal(data1, data2)

    def test_generate_user_demographics(self, sample_simulator):
        """Test user demographics generation."""
        df = sample_simulator.generate_user_demographics()

        assert len(df) == 50
        assert "user_id" in df.columns
        assert "age" in df.columns
        assert "income_bracket" in df.columns
        assert "device_type" in df.columns
        assert "user_type" in df.columns
        assert "region" in df.columns
        assert "tech_savviness" in df.columns

        # Check value ranges
        assert df["age"].min() >= 18
        assert df["age"].max() <= 70
        assert df["tech_savviness"].min() >= 0
        assert df["tech_savviness"].max() <= 100

        # Check categorical values
        assert df["income_bracket"].isin(["<30k", "30k-50k", "50k-75k"]).all()
        assert df["device_type"].isin(["mobile", "desktop"]).all()
        assert df["user_type"].isin(["new", "returning"]).all()
        assert df["region"].isin(["West", "East"]).all()

    def test_generate_complete_dataset(self, sample_simulator):
        """Test complete dataset generation."""
        df = sample_simulator.generate_complete_dataset()

        # Check shape
        assert len(df) == 50
        assert len(df.columns) >= 15  # Should have many columns

        # Check key columns exist
        required_columns = [
            "user_id",
            "age",
            "income_bracket",
            "device_type",
            "user_type",
            "region",
            "tech_savviness",
            "filed_2023",
            "filed_2024",
            "used_smart_assistant",
            "time_to_complete_2023",
            "time_to_complete_2024",
            "sessions_2023",
            "sessions_2024",
            "support_tickets_2023",
            "support_tickets_2024",
            "satisfaction_2024",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Check derived features
        if "time_improvement" in df.columns:
            assert (
                df["time_improvement"]
                == df["time_to_complete_2023"] - df["time_to_complete_2024"]
            ).all()

        if "session_improvement" in df.columns:
            assert (
                df["session_improvement"] == df["sessions_2023"] - df["sessions_2024"]
            ).all()

        # Check early login column (it may be named early_login_2024)
        early_login_cols = [col for col in df.columns if "early_login" in col]
        assert len(early_login_cols) > 0, (
            f"No early login column found in: {list(df.columns)}"
        )

    def test_treatment_effect_direction(self, sample_simulator):
        """Test that treatment effects go in expected direction."""
        df = sample_simulator.generate_complete_dataset()

        treated = df[df["used_smart_assistant"] == 1]
        control = df[df["used_smart_assistant"] == 0]

        if len(treated) > 5 and len(control) > 5:  # Need sufficient sample size
            # Treated users should file more on average
            assert treated["filed_2024"].mean() >= control["filed_2024"].mean() - 0.1

            # Treated users should have higher satisfaction on average
            assert (
                treated["satisfaction_2024"].mean()
                >= control["satisfaction_2024"].mean() - 0.5
            )


class TestGenerateAndSaveData:
    """Test the convenience function for data generation."""

    def test_generate_and_save_data_creates_file(self, real_config_path):
        """Test that generate_and_save_data creates a file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            df = generate_and_save_data(
                output_path=output_path, n_users=100, config_path=real_config_path
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 100
            assert Path(output_path).exists()

            # Load saved file and verify
            saved_df = pd.read_csv(output_path)
            assert len(saved_df) == 100

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_and_save_data_default_path(self, real_config_path):
        """Test generate_and_save_data with default path."""
        # Clean up any existing file
        default_path = Path("data/simulated_users.csv")
        backup_exists = default_path.exists()
        if backup_exists:
            backup_path = Path("data/simulated_users_backup.csv")
            default_path.rename(backup_path)

        try:
            df = generate_and_save_data(n_users=50, config_path=real_config_path)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
            assert default_path.exists()

        finally:
            # Restore backup if it existed
            if backup_exists:
                default_path.unlink(missing_ok=True)
                backup_path.rename(default_path)
            elif default_path.exists():
                default_path.unlink()


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_probability_weights(self, sample_config):
        """Test that invalid probability weights are handled."""
        # Weights that don't sum to 1 should raise error
        sample_config["demographics"]["income_brackets"]["weights"] = [0.5, 0.3, 0.3]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name

        try:
            simulator = TaxSoftwareDataSimulator(n_users=10, config_path=temp_path)
            # Should raise ValueError because weights don't sum to 1.0
            with pytest.raises(ValueError, match="probabilities do not sum to 1"):
                df = simulator.generate_complete_dataset()
        finally:
            Path(temp_path).unlink()

    def test_extreme_parameters(self, sample_config):
        """Test with extreme parameter values."""
        # Very high treatment adoption
        sample_config["treatment"]["base_adoption_rate"] = 0.95
        sample_config["treatment"]["min_probability"] = 0.9

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name

        try:
            simulator = TaxSoftwareDataSimulator(n_users=20, config_path=temp_path)
            df = simulator.generate_complete_dataset()

            # Should have very high treatment rate
            treatment_rate = df["used_smart_assistant"].mean()
            assert treatment_rate >= 0.8

        finally:
            Path(temp_path).unlink()


class TestDataConsistency:
    """Test data consistency across generation."""

    def test_boolean_columns_are_binary(self, sample_simulator):
        """Test that boolean columns contain only 0 and 1."""
        df = sample_simulator.generate_complete_dataset()

        # Identify boolean columns
        bool_columns = []
        for col in df.columns:
            if (
                "filed" in col
                or "used_smart_assistant" in col
                or "support_tickets" in col
                or "early_login" in col
            ):
                bool_columns.append(col)

        for col in bool_columns:
            if col in df.columns:
                unique_vals = set(df[col].unique())
                assert unique_vals.issubset({0, 1}), (
                    f"{col} contains non-binary values: {unique_vals}"
                )

    def test_positive_numeric_columns(self, sample_simulator):
        """Test that numeric columns that should be positive are positive."""
        df = sample_simulator.generate_complete_dataset()

        # Test satisfaction scores
        satisfaction_columns = [col for col in df.columns if "satisfaction" in col]
        for col in satisfaction_columns:
            assert df[col].min() >= 1, f"{col} should be >= 1, got min: {df[col].min()}"
            assert df[col].max() <= 10, (
                f"{col} should be <= 10, got max: {df[col].max()}"
            )

        # Test sessions (should be at least 1 for users who filed)
        session_columns = [col for col in df.columns if "sessions" in col]
        for col in session_columns:
            filed_col = col.replace("sessions", "filed")
            if filed_col in df.columns:
                # For users who filed, sessions should be >= 1
                filed_users = df[df[filed_col] == 1]
                if len(filed_users) > 0:
                    assert filed_users[col].min() >= 1, (
                        f"{col} should be >= 1 for users who filed"
                    )

        # Test time_to_complete (should be > 0 for users who filed)
        time_columns = [col for col in df.columns if "time_to_complete" in col]
        for col in time_columns:
            filed_col = col.replace("time_to_complete", "filed")
            if filed_col in df.columns:
                # For users who filed, time should be > 0
                filed_users = df[df[filed_col] == 1]
                if len(filed_users) > 0:
                    assert filed_users[col].min() > 0, (
                        f"{col} should be > 0 for users who filed"
                    )

                # For users who didn't file, time should be 0
                not_filed_users = df[df[filed_col] == 0]
                if len(not_filed_users) > 0:
                    assert (not_filed_users[col] == 0).all(), (
                        f"{col} should be 0 for users who didn't file"
                    )

    def test_derived_features_calculation(self, sample_simulator):
        """Test that derived features are calculated correctly."""
        df = sample_simulator.generate_complete_dataset()

        # Test time improvement calculation
        if all(
            col in df.columns
            for col in [
                "time_improvement",
                "time_to_complete_2023",
                "time_to_complete_2024",
            ]
        ):
            expected_time_improvement = (
                df["time_to_complete_2023"] - df["time_to_complete_2024"]
            )
            pd.testing.assert_series_equal(
                df["time_improvement"], expected_time_improvement, check_names=False
            )

        # Test session improvement calculation
        if all(
            col in df.columns
            for col in ["session_improvement", "sessions_2023", "sessions_2024"]
        ):
            expected_session_improvement = df["sessions_2023"] - df["sessions_2024"]
            pd.testing.assert_series_equal(
                df["session_improvement"],
                expected_session_improvement,
                check_names=False,
            )
