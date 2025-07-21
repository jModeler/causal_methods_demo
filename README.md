# Causal Methods Demo

A comprehensive demonstration of causal inference methods for analyzing the impact of digital interventions, specifically designed around a tax software Smart Filing Assistant use case.

## 🎯 Overview

This project implements and demonstrates various causal inference techniques to evaluate the effectiveness of a Smart Filing Assistant feature in tax preparation software. The methods help answer key business questions like "What is the true causal effect of our new feature on user conversion rates?"

## 🏗️ Project Structure

```
causal_methods_demo/
├── src/
│   ├── causal_methods/
│   │   ├── did.py              # Difference-in-Differences implementation
│   │   └── psm.py              # Propensity Score Matching implementation
│   └── data_simulation.py      # Synthetic data generation
├── notebooks/
│   ├── 01_did_tax.ipynb       # DiD demonstration notebook
│   └── 02_psm_tax.ipynb       # PSM demonstration notebook
├── tests/                      # Comprehensive test suite
├── config/                     # Configuration files
└── docs/                       # Documentation
```

## 🔬 Implemented Methods

### 1. **Propensity Score Matching (PSM)** ✨ *New!*
- **Purpose**: Reduce selection bias by matching treated and control units based on propensity scores
- **Implementation**: `src/causal_methods/psm.py`
- **Notebook**: `notebooks/02_psm_tax.ipynb`
- **Features**:
  - Multiple matching algorithms (nearest neighbor, caliper)
  - Automated covariate balance assessment
  - Statistical significance testing with proper binary outcome handling
  - Rich visualizations (propensity distributions, balance plots, effect plots)
  - Robust error handling and edge case management

### 2. **Difference-in-Differences (DiD)**
- **Purpose**: Estimate treatment effects using temporal variation
- **Implementation**: `src/causal_methods/did.py`
- **Notebook**: `notebooks/01_did_tax.ipynb`
- **Features**:
  - Panel data preparation and regression analysis
  - Parallel trends assumption testing
  - Treatment effect visualization

### 3. **Synthetic Data Generation**
- **Purpose**: Generate realistic tax software user data for method demonstration
- **Implementation**: `src/data_simulation.py`
- **Features**:
  - Configurable user demographics and behavior patterns
  - Realistic selection bias and confounding factors
  - Multiple scenario configurations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd causal_methods_demo

# Install dependencies using uv
uv sync
```

### Basic Usage

#### Propensity Score Matching Example
```python
from src.causal_methods.psm import PropensityScoreMatching, load_and_analyze_psm

# Quick analysis using convenience function
psm = load_and_analyze_psm(
    file_path="data/your_data.csv",
    treatment_col="used_smart_assistant",
    outcome_cols="filed_2024",
    matching_method="nearest_neighbor",
    caliper=0.1
)

# View results
print(psm.generate_summary_report())
```

#### Manual PSM Workflow
```python
from src.causal_methods.psm import PropensityScoreMatching

# Initialize PSM
psm = PropensityScoreMatching(your_dataframe)

# Estimate propensity scores
ps_results = psm.estimate_propensity_scores(
    covariates=['age', 'income', 'tech_savviness']
)

# Perform matching
matching_results = psm.perform_matching(method='nearest_neighbor', caliper=0.1)

# Assess balance
balance_results = psm.assess_balance()

# Estimate treatment effects
effects = psm.estimate_treatment_effects(outcome_cols='your_outcome')
```

#### Difference-in-Differences Example
```python
from src.causal_methods.did import DifferenceInDifferences

# Initialize DiD
did = DifferenceInDifferences(your_dataframe)

# Prepare panel data
panel_df = did.prepare_panel_data()

# Estimate treatment effects
results = did.estimate_did()
```

### Data Generation
```python
from src.data_simulation import generate_and_save_data

# Generate synthetic tax software data
df = generate_and_save_data(
    output_path="data/synthetic_data.csv",
    n_users=1000,
    config_path="config/simulation_config.yaml"
)
```

## 📊 Example Scenarios

The project includes several pre-configured scenarios demonstrating different business contexts:

- **Baseline**: Standard adoption and treatment effects
- **High Treatment**: Increased treatment effect scenario
- **Low Adoption**: Lower treatment adoption rates

Configure scenarios using YAML files in the `config/` directory.

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific method tests
uv run pytest tests/test_psm.py -v          # PSM tests
uv run pytest tests/test_did.py -v          # DiD tests
uv run pytest tests/test_integration.py -v  # Integration tests
```

**Test Coverage**: 84% overall with 97 tests passing

## 📚 Documentation

- **[Configuration Parameters](docs/configuration_parameters.md)**: Detailed explanation of all simulation parameters
- **[PSM Notebook](notebooks/02_psm_tax.ipynb)**: Step-by-step PSM tutorial
- **[DiD Notebook](notebooks/01_did_tax.ipynb)**: Difference-in-differences demonstration
- **[API Documentation](docs/)**: Detailed API reference

## 🎯 Key Features

### Statistical Rigor
- **Proper statistical tests**: Automatic selection of appropriate tests for different data types
- **Binary outcome handling**: Special treatment for boolean/binary outcomes with proportion tests
- **Robust error handling**: Conservative fallbacks when statistical tests fail
- **Comprehensive diagnostics**: Balance assessment, model fit statistics, effect visualization

### Production Ready
- **Comprehensive testing**: 97 tests with 84% coverage
- **Clean code**: Ruff linting, consistent formatting, type hints
- **Error handling**: Graceful handling of edge cases and invalid inputs
- **Documentation**: Detailed docstrings and user guides

### Flexibility
- **Multiple matching methods**: Nearest neighbor, caliper, with/without replacement
- **Configurable parameters**: Extensive customization options
- **Modular design**: Easy to extend with additional methods
- **Integration ready**: Consistent APIs across different causal methods

## 🔧 Development

### Code Quality
```bash
# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/
```

### Adding New Methods
1. Create implementation in `src/causal_methods/your_method.py`
2. Add comprehensive tests in `tests/test_your_method.py`
3. Create demonstration notebook in `notebooks/`
4. Update documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is properly formatted
5. Update documentation
6. Submit a pull request

## 📈 Roadmap

**Phase 2** (Planned):
- [ ] Double Machine Learning (DML)
- [ ] Instrumental Variables (IV)
- [ ] CUPED (Controlled-experiment Using Pre-Experiment Data)
- [ ] Causal Forests
- [ ] Synthetic Control Methods

**Phase 3** (Future):
- [ ] Web interface for interactive analysis
- [ ] Automated report generation
- [ ] Integration with common data platforms
- [ ] Advanced sensitivity analysis tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python tooling (uv, ruff, pytest)
- Inspired by real-world causal inference challenges in product analytics
- Designed for educational and practical use in business contexts 