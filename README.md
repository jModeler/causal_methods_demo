# Causal Inference Methods for Tax Software Impact Analysis

This repository demonstrates various causal inference methods to estimate the impact of a **"Smart Filing Assistant"** feature on user conversion and engagement in a B2C tax software company.

## 📊 Business Context

**Scenario**: A tax software company launched a new "Smart Filing Assistant" feature globally before the 2024 tax season. Since no A/B test was conducted, we need to use observational data and causal inference methods to estimate the feature's true impact.

**Key Questions**:
- What is the causal effect of the Smart Filing Assistant on user conversion rates?
- How does the feature impact user engagement metrics (time to complete, support tickets)?
- Are there heterogeneous treatment effects across different user segments?

## 🎯 Methodology

We demonstrate several causal inference methods:

1. **Difference-in-Differences (DiD)** - Using 2023 as baseline
2. **Propensity Score Matching (PSM)** - Balancing treatment and control groups
3. **Double Machine Learning (DML)** - Robust estimation with ML models
4. **CUPED** - Variance reduction using pre-treatment outcomes
5. **Instrumental Variables (IV)** - Addressing unobserved confounders
6. **Causal Forests** - Heterogeneous treatment effects

## 🗂️ Repository Structure

```
causal-methods-demo/
├── README.md
├── pyproject.toml
├── config/
│   ├── simulation_config.yaml         # Main configuration file
│   ├── scenario_high_treatment.yaml   # High treatment effect scenario
│   ├── scenario_low_adoption.yaml     # Low adoption rate scenario
│   └── README.md                      # Configuration system overview
├── data/
│   └── simulated_users.csv
├── docs/
│   ├── README.md                      # Documentation index
│   └── configuration_parameters.md    # Complete parameter reference
├── notebooks/
│   ├── 00_generate_data.ipynb
│   ├── 01_did_tax.ipynb
│   ├── 02_psm_tax.ipynb
│   ├── 03_double_ml_tax.ipynb
│   ├── 04_cuped_tax.ipynb
│   ├── 05_iv_tax.ipynb
│   └── 06_causal_forest.ipynb
└── src/
    ├── data_simulation.py
    └── causal_methods/
        ├── did.py
        ├── psm.py
        ├── dml.py
        ├── cuped.py
        ├── iv.py
        └── causal_forest.py
```

## 📈 Synthetic Dataset

The dataset includes:

### User Demographics
- Age, income bracket, device type, geographic region
- User type (new vs. returning)
- Tech-savviness score (hidden confounder)

### Pre-treatment (2023 Baseline)
- Filing behavior and completion metrics
- Support ticket history
- Early adoption indicators

### Treatment
- Smart Filing Assistant usage (selection bias present)

### Post-treatment (2024 Outcomes)
- Conversion rates (filed taxes)
- Completion time and number of sessions
- Support tickets and satisfaction scores

## ⚙️ Configuration System

**NEW**: All simulation parameters are now configurable via YAML files!

### Configuration Files
- **`config/simulation_config.yaml`** - Main configuration with all parameters
- **`config/scenario_high_treatment.yaml`** - High treatment effect scenario
- **`config/scenario_low_adoption.yaml`** - Low adoption rate scenario

### Documentation
- **`docs/configuration_parameters.md`** - Complete parameter reference guide
- **`config/README.md`** - Configuration system overview
- **`docs/README.md`** - Documentation index and navigation

### Benefits
- 🚫 **No magic numbers** - All parameters externalized
- 🔬 **Easy experimentation** - Change parameters without code changes
- 📊 **Multiple scenarios** - Compare different assumptions
- 📝 **Self-documenting** - YAML files explain each parameter

### Configuration Categories
- **Demographics**: Age, income, device, region distributions
- **Tech-savviness**: Scoring algorithm and adjustments
- **Treatment Assignment**: Adoption probability factors
- **Baseline Behavior**: 2023 filing patterns
- **Outcomes**: 2024 treatment effects and metrics

### Usage Examples
```python
# Use default configuration
from src.data_simulation import generate_and_save_data
df = generate_and_save_data()

# Use custom configuration
df = generate_and_save_data(config_path="config/scenario_high_treatment.yaml")

# Generate multiple scenarios
from src.data_simulation import TaxSoftwareDataSimulator
simulator = TaxSoftwareDataSimulator(
    n_users=5000, 
    config_path="config/scenario_low_adoption.yaml"
)
df = simulator.generate_complete_dataset()
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd causal-methods-demo
   ```

2. **Set up the environment with uv**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Generate synthetic data**:
   ```bash
   python src/data_simulation.py
   ```

5. **Launch Jupyter notebooks**:
   ```bash
   jupyter notebook
   ```

### Alternative: Development Installation

To install with development dependencies:
```bash
uv sync --extra dev
```

## 📚 Notebooks Guide

### [00_generate_data.ipynb](notebooks/00_generate_data.ipynb)
- **Purpose**: Generate and explore synthetic dataset
- **Key Insights**: Selection bias visualization, outcome distributions
- **Runtime**: ~2 minutes

### [01_did_tax.ipynb](notebooks/01_did_tax.ipynb)
- **Method**: Difference-in-Differences
- **Assumptions**: Parallel trends, no spillovers
- **Use Case**: When you have pre/post treatment periods

### [02_psm_tax.ipynb](notebooks/02_psm_tax.ipynb)
- **Method**: Propensity Score Matching
- **Assumptions**: Unconfoundedness, overlap
- **Use Case**: Rich set of observed confounders

### [03_double_ml_tax.ipynb](notebooks/03_double_ml_tax.ipynb)
- **Method**: Double Machine Learning
- **Assumptions**: Unconfoundedness
- **Use Case**: High-dimensional confounders, robust estimation

### [04_cuped_tax.ipynb](notebooks/04_cuped_tax.ipynb)
- **Method**: CUPED (Controlled-experiment Using Pre-Experiment Data)
- **Assumptions**: Linear relationship with pre-treatment outcomes
- **Use Case**: Variance reduction when you have baseline metrics

### [05_iv_tax.ipynb](notebooks/05_iv_tax.ipynb)
- **Method**: Instrumental Variables
- **Assumptions**: Relevance, exclusion restriction, independence
- **Use Case**: Unobserved confounders present

### [06_causal_forest.ipynb](notebooks/06_causal_forest.ipynb)
- **Method**: Causal Forests
- **Assumptions**: Unconfoundedness
- **Use Case**: Heterogeneous treatment effects across subgroups

## 🔍 Key Findings

### Selection Bias Identified
- Tech-savvy users more likely to adopt Smart Filing Assistant
- Higher-income and younger users show higher adoption rates
- Returning users and early adopters more likely to use the feature

### Treatment Effects
- **Naive estimate**: ~12% increase in filing rates
- **Causal estimates** (after controlling for confounders): ~8% increase
- **Heterogeneous effects**: Stronger benefits for older and less tech-savvy users
- **Secondary outcomes**: Reduced completion time and support tickets

## 🛠️ Development

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy .

# Run tests
uv run pytest
```

### Adding New Methods

1. Create a new module in `src/causal_methods/`
2. Implement the estimation method with a consistent API
3. Create a corresponding notebook demonstrating the method
4. Update this README with method description

## 📖 Causal Inference Resources

### Books
- "Causal Inference: The Mixtape" by Scott Cunningham
- "Causal Inference for the Brave and True" by Matheus Facure
- "Mostly Harmless Econometrics" by Angrist & Pischke

### Libraries Used
- **[EconML](https://github.com/microsoft/EconML)**: Microsoft's causal ML library
- **[DoWhy](https://github.com/microsoft/dowhy)**: Microsoft's causal inference library  
- **[CausalML](https://github.com/uber/causalml)**: Uber's causal ML library
- **[LinearModels](https://github.com/bashtage/linearmodels)**: IV and panel data methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Commit changes (`git commit -am 'Add new causal method'`)
4. Push to branch (`git push origin feature/new-method`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Questions?

If you have questions about the implementation or want to discuss causal inference methods, please:
- Open an issue for bugs or feature requests
- Start a discussion for methodology questions
- Check the notebook comments for detailed explanations

---

**Disclaimer**: This is a demonstration project using synthetic data. Real-world causal inference requires careful consideration of assumptions and domain expertise. 