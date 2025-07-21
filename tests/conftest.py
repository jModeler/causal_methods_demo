"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_simulation import TaxSoftwareDataSimulator


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "simulation": {"random_seed": 42, "default_n_users": 100},
        "demographics": {
            "income_brackets": {
                "values": ["<30k", "30k-50k", "50k-75k"],
                "weights": [0.4, 0.4, 0.2],
            },
            "device_types": {"values": ["mobile", "desktop"], "weights": [0.6, 0.4]},
            "user_types": {"values": ["new", "returning"], "weights": [0.3, 0.7]},
            "regions": {"values": ["West", "East"], "weights": [0.5, 0.5]},
            "age": {"mean": 40, "std": 12, "min_age": 18, "max_age": 70},
        },
        "tech_savviness": {
            "base_score": 50,
            "age_adjustments": {
                "young_threshold": 35,
                "young_boost": 15,
                "old_threshold": 55,
                "old_penalty": -10,
            },
            "income_adjustments": {
                "high_income_brackets": ["50k-75k"],
                "high_income_boost": 10,
                "low_income_brackets": ["<30k"],
                "low_income_penalty": -5,
            },
            "region_adjustments": {"west_boost": 5, "midwest_south_penalty": -3},
            "std": 12,
            "min_score": 0,
            "max_score": 100,
        },
        "baseline_2023": {
            "filing": {
                "base_rate": 0.7,
                "income_effects": {
                    "high_income_boost": 0.1,
                    "low_income_penalty": -0.1,
                },
                "returning_user_boost": 0.05,
                "prime_age_min": 25,
                "prime_age_max": 55,
                "prime_age_boost": 0.03,
                "non_prime_penalty": -0.03,
            },
            "time_to_complete": {
                "base_time": 120,
                "returning_user_reduction": -20,
                "high_tech_reduction": -15,
                "low_tech_penalty": 30,
                "std": 25,
                "min_time": 30,
            },
            "sessions": {
                "base_sessions": 2.0,
                "mobile_penalty": 0.3,
                "low_tech_penalty": 0.5,
            },
            "support_tickets": {
                "base_rate": 0.1,
                "low_tech_boost": 0.1,
                "elderly_boost": 0.05,
            },
        },
        "early_login": {
            "base_probability": 0.25,
            "high_tech_boost": 0.15,
            "returning_user_boost": 0.1,
        },
        "treatment": {
            "base_adoption_rate": 0.4,
            "tech_effects": {
                "high_threshold": 70,
                "high_boost": 0.2,
                "medium_threshold": 50,
                "medium_boost": 0.08,
                "low_threshold": 30,
                "low_penalty": -0.15,
            },
            "age_effects": {
                "young_threshold": 35,
                "young_boost": 0.1,
                "old_threshold": 55,
                "old_penalty": -0.08,
            },
            "device_effects": {"mobile_boost": 0.03, "tablet_penalty": -0.03},
            "early_login_boost": 0.15,
            "returning_user_boost": 0.03,
            "income_effects": {"high_income_boost": 0.08, "low_income_penalty": -0.03},
            "min_probability": 0.05,
            "max_probability": 0.9,
        },
        "outcomes_2024": {
            "filing": {
                "base_rate": 0.68,
                "treatment_effects": {
                    "base_effect": 0.08,
                    "low_tech_boost": 0.04,
                    "high_tech_reduction": -0.01,
                    "older_user_boost": 0.02,
                    "new_user_boost": 0.03,
                },
                "filed_2023_boost": 0.12,
                "apply_demographic_effects": True,
                "max_probability": 0.95,
            },
            "time_to_complete": {
                "treatment_time_reduction": {
                    "min_reduction": 0.15,
                    "max_reduction": 0.25,
                },
                "std": 20,
                "min_time": 20,
            },
            "sessions": {
                "treatment_session_reduction": {
                    "min_reduction": 0.1,
                    "max_reduction": 0.3,
                }
            },
            "support_tickets": {
                "base_rate": 0.08,
                "filed_2023_penalty": 0.03,
                "treatment_reduction": 0.6,
            },
            "satisfaction": {
                "base_score": 7.0,
                "treatment_boost": 0.6,
                "high_tech_boost": 0.2,
                "support_history_penalty": -0.3,
                "std": 1.0,
                "min_score": 1,
                "max_score": 10,
            },
        },
        "output": {
            "default_path": "data/test_users.csv",
            "include_derived_features": True,
            "derived_features": ["time_improvement", "session_improvement"],
        },
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def sample_simulator(temp_config_file):
    """Create a sample data simulator for testing."""
    return TaxSoftwareDataSimulator(n_users=50, config_path=temp_config_file)


@pytest.fixture
def sample_dataset(sample_simulator):
    """Generate a small sample dataset for testing."""
    return sample_simulator.generate_complete_dataset()


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for DiD testing."""
    np.random.seed(42)
    n_users = 100

    # Create sample data
    data = {
        "user_id": range(n_users),
        "age": np.random.normal(40, 12, n_users),
        "income_bracket": np.random.choice(["<30k", "30k-50k", "50k-75k"], n_users),
        "device_type": np.random.choice(["mobile", "desktop"], n_users),
        "used_smart_assistant": np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
        "filed_2023": np.random.choice([0, 1], n_users, p=[0.3, 0.7]),
        "filed_2024": np.random.choice([0, 1], n_users, p=[0.25, 0.75]),
        "time_to_complete_2023": np.random.gamma(2, 60, n_users),
        "time_to_complete_2024": np.random.gamma(2, 50, n_users),
        "sessions_2023": np.random.poisson(2.5, n_users),
        "sessions_2024": np.random.poisson(2.0, n_users),
        "support_tickets_2023": np.random.choice([0, 1], n_users, p=[0.85, 0.15]),
        "tech_savviness": np.random.normal(50, 15, n_users),
    }

    return pd.DataFrame(data)


@pytest.fixture
def real_config_path():
    """Path to the real simulation config for integration tests."""
    return "config/simulation_config.yaml"
