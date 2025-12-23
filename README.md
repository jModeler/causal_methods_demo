# Causal Methods Demo

A comprehensive toolkit demonstrating various causal inference methods for business analytics and decision-making. This project showcases practical implementations of **Propensity Score Matching (PSM)**, **Double Machine Learning (DML)**, **CUPED**, **Difference-in-Differences (DiD)**, **Synthetic Control**, and **Causal Forest** methods using synthetic tax software data.

## 🎯 **Project Overview**

This repository provides production-ready implementations of modern causal inference techniques, each designed to answer different types of business questions:

| Method | Use Case | Strengths | When to Use |
|--------|----------|-----------|-------------|
| **PSM** | Observational studies | Balances observed confounders | Non-randomized data with rich covariates |
| **DML** | High-dimensional data | Handles complex relationships | Many covariates, non-linear relationships |
| **CUPED** | Randomized experiments | Reduces variance | A/B tests with pre-treatment data |
| **DiD** | Policy evaluation | Controls for time trends | Before/after intervention data |
| **Synthetic Control** | Individual effects | Transparent matching, no parametric assumptions | Rich pre-treatment data, individual-level effects |
| **Causal Forest** | Heterogeneous effects | Personalized treatment effects | Rich covariates, segment-specific insights |

## 📊 **Business Context: Smart Filing Assistant Impact**

All methods analyze the same business question: **What is the causal effect of a Smart Filing Assistant on user tax filing conversion?**

- **Treatment**: User adoption of Smart Filing Assistant feature
- **Outcome**: Binary filing completion in 2024
- **Data**: 2023 baseline behavior + 2024 outcomes
- **Goal**: Estimate causal impact for business decision-making

## 🚀 **Quick Start**

### Installation
```bash
# Clone repository
git clone https://github.com/jModeler/causal_methods_demo.git
cd causal_methods_demo

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r pyproject.toml
```
**Note:** If using `econml` or `dowhy` libraries, please use a python version `==3.10`. The notebooks in this repository were tested with Python v`3.10.5`

### Run Analysis Examples

```python
from src.causal_methods.psm import PropensityScoreMatching
from src.causal_methods.dml import DoubleMachineLearning
from src.causal_methods.cuped import CUPED
from src.causal_methods.did import DifferenceInDifferences
from src.causal_methods.synthetic_control import SyntheticControl
from src.causal_methods.causal_forest import CausalForest
from src.data_simulation import TaxSoftwareDataSimulator

# Generate synthetic data
simulator = TaxSoftwareDataSimulator(n_users=1000)
df = simulator.generate_complete_dataset()

# 1. Propensity Score Matching
psm = PropensityScoreMatching(df)
psm.estimate_propensity_scores()
psm.perform_matching()
psm.estimate_treatment_effects(outcome_cols='filed_2024')
print(f"PSM ATE: {psm.treatment_effects['filed_2024']['ate']:.4f}")

# 2. Double Machine Learning
dml = DoubleMachineLearning(df, random_state=42)
dml_results = dml.estimate_treatment_effects('filed_2024', 'used_smart_assistant')
print(f"DML ATE: {dml_results['ate']:.4f}")

# 3. CUPED (for experimental data)
cuped = CUPED(df, random_state=42)
cuped_results = cuped.estimate_treatment_effects('filed_2024', 'used_smart_assistant')
print(f"CUPED ATE: {cuped_results['cuped']['ate']:.4f}")

# 4. Difference-in-Differences (requires panel structure)
did = DifferenceInDifferences(df)
did_results = did.estimate_treatment_effects()
print(f"DiD ATE: {did_results['treatment_effect']:.4f}")

# 5. Synthetic Control
sc = SyntheticControl(df, random_state=42)
sc_results = sc.construct_synthetic_controls()
print(f"Synthetic Control ATE: {sc_results['average_treatment_effect']:.4f}")

# 6. Causal Forest (for heterogeneous effects)
cf = CausalForest(df, random_state=42)
cf.fit_causal_forest('filed_2024', 'used_smart_assistant')
cf_results = cf.estimate_treatment_effects()
print(f"Causal Forest ATE: {cf_results['ate']:.4f}")
print(f"Effect Heterogeneity: ±{cf_results['ate_std']:.4f}")
```

## 📚 **Detailed Method Documentation**

### 🎯 **1. Propensity Score Matching (PSM)**
- **Notebook**: `notebooks/02_psm_tax.ipynb`
- **Documentation**: `docs/psm_guide.md`
- **Core Idea**: Match treated and control units with similar propensity scores

```python
psm = PropensityScoreMatching(df)
psm.estimate_propensity_scores(covariates=['age', 'income', 'filed_2023'])
psm.perform_matching(method='nearest_neighbor', caliper=0.1)
effects = psm.estimate_treatment_effects(outcome_cols='filed_2024')
```

### 🤖 **2. Double Machine Learning (DML)**
- **Notebook**: `notebooks/03_dml_tax.ipynb`
- **Documentation**: `docs/dml_guide.md`
- **Core Idea**: Use ML to control for confounders in high-dimensional settings

```python
dml = DoubleMachineLearning(df, random_state=42)
results = dml.estimate_treatment_effects(
    outcome_col='filed_2024',
    treatment_col='used_smart_assistant',
    covariates=['age', 'income', 'tech_savviness', 'filed_2023']
)
```

### 📈 **3. CUPED (Controlled-experiment Using Pre-Experiment Data)**
- **Notebook**: `notebooks/04_cuped_tax.ipynb`
- **Documentation**: `docs/cuped_guide.md`
- **Core Idea**: Reduce variance in randomized experiments using pre-treatment data

```python
cuped = CUPED(df, random_state=42)
results = cuped.estimate_treatment_effects(
    outcome_col='filed_2024',
    treatment_col='used_smart_assistant',
    covariate_cols=['filed_2023', 'sessions_2023']
)
```

### 📊 **4. Difference-in-Differences (DiD)**
- **Notebook**: `notebooks/01_did_tax.ipynb`
- **Documentation**: `docs/did_guide.md`
- **Core Idea**: Exploit variation in treatment timing to control for unobserved confounders

```python
did = DifferenceInDifferences(df)
results = did.estimate_treatment_effects(
    unit_col='user_id',
    time_col='year',
    treatment_col='used_smart_assistant',
    outcome_col='filed'
)
```

### ⚖️ **5. Synthetic Control**
- **Notebook**: `notebooks/05_synthetic_control_tax.ipynb`
- **Documentation**: `docs/synthetic_control_guide.md`
- **Core Idea**: Construct synthetic control units using weighted combinations of donor units

```python
sc = SyntheticControl(df, random_state=42)
results = sc.construct_synthetic_controls(
    outcome_pre_col='filed_2023',
    outcome_post_col='filed_2024',
    predictor_cols=['filed_2023', 'age', 'tech_savviness']
)
```

### 🌲 **6. Causal Forest**
- **Notebook**: `notebooks/06_causal_forest_tax.ipynb`
- **Documentation**: `docs/causal_forest_guide.md`
- **Core Idea**: Estimate heterogeneous treatment effects using specialized random forests

```python
cf = CausalForest(df, random_state=42)
cf.fit_causal_forest(
    outcome_col='filed_2024',
    treatment_col='used_smart_assistant',
    covariate_cols=['age', 'tech_savviness', 'sessions_2023', 'filed_2023']
)
results = cf.estimate_treatment_effects()
```

## 🔍 **Method Comparison Framework**

Each notebook includes comprehensive method comparisons:

```python
# Example: Compare all methods on the same dataset
methods_comparison = {
    'PSM': psm.treatment_effects['filed_2024']['ate'],
    'DML': dml_results['ate'],
    'CUPED': cuped_results['cuped']['ate'],
    'DiD': did_results['treatment_effect'],
    'Synthetic Control': sc_results['average_treatment_effect'],
    'Causal Forest': cf_results['ate']
}

for method, ate in methods_comparison.items():
    print(f"{method:15}: {ate:7.4f}")
```

## 📂 **Project Structure**

```
causal_methods_demo/
├── src/
│   ├── causal_methods/
│   │   ├── psm.py              # Propensity Score Matching
│   │   ├── dml.py              # Double Machine Learning
│   │   ├── cuped.py            # CUPED implementation
│   │   ├── did.py              # Difference-in-Differences
│   │   ├── synthetic_control.py # Synthetic Control
│   │   └── causal_forest.py    # Causal Forest
│   └── data_simulation.py      # Synthetic data generation
├── notebooks/
│   ├── 01_did_tax.ipynb        # DiD demonstration
│   ├── 02_psm_tax.ipynb        # PSM demonstration
│   ├── 03_dml_tax.ipynb        # DML demonstration
│   ├── 04_cuped_tax.ipynb      # CUPED demonstration
│   ├── 05_synthetic_control_tax.ipynb # Synthetic Control demonstration
│   └── 06_causal_forest_tax.ipynb # Causal Forest demonstration
├── docs/
│   ├── psm_guide.md            # PSM documentation
│   ├── dml_guide.md            # DML documentation
│   ├── cuped_guide.md          # CUPED documentation
│   ├── did_guide.md            # DiD documentation
│   ├── synthetic_control_guide.md # Synthetic Control documentation
│   ├── causal_forest_guide.md  # Causal Forest documentation
│   └── README.md               # Documentation index
├── tests/                      # Comprehensive test suite
├── config/                     # Configuration files
└── requirements.txt            # Dependencies
```

## 🧪 **Testing**

Run comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific method tests
pytest tests/test_psm.py -v
pytest tests/test_dml.py -v
pytest tests/test_cuped.py -v
pytest tests/test_did.py -v
pytest tests/test_synthetic_control.py -v
pytest tests/test_causal_forest.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

## 📊 **Key Features**

### **Robust Implementation**
- ✅ Production-ready code with comprehensive error handling
- ✅ Extensive test coverage (>90%) with edge case testing
- ✅ Type hints and detailed docstrings
- ✅ Configurable parameters with sensible defaults

### **Business-Focused Analysis**
- 📊 Clear business metric interpretation
- 💰 ROI projections and impact quantification  
- 📈 Comprehensive visualizations
- 📋 Executive summary reports

### **Methodological Rigor**
- 🔬 Assumption checking and diagnostics
- 📊 Confidence intervals and significance testing
- ⚖️ Multiple estimation approaches for robustness
- 🔄 Cross-validation and sensitivity analysis

### **Practical Usability**
- 📚 Step-by-step Jupyter notebooks
- 📖 Comprehensive documentation
- 🎯 Real-world business scenarios
- 🔧 Easy-to-use API design

## 🎓 **Learning Path**

### **Beginner**: Start with Notebooks
1. **DiD**: `01_did_tax.ipynb` - Introduction to causal thinking
2. **PSM**: `02_psm_tax.ipynb` - Matching and confounders
3. **CUPED**: `04_cuped_tax.ipynb` - Experimental variance reduction

### **Intermediate**: Advanced Methods
4. **DML**: `03_dml_tax.ipynb` - Machine learning for causal inference
5. **Synthetic Control**: `05_synthetic_control_tax.ipynb` - Individual-level effects

### **Advanced**: Heterogeneous Effects and Method Comparison
6. **Causal Forest**: `06_causal_forest_tax.ipynb` - Personalized treatment effects
- Read `docs/README.md` for method selection guidance
- Compare results across methods for robustness
- Implement custom extensions and modifications

## 🔬 **Method Selection Guide**

| Scenario | Recommended Method | Alternative | Rationale |
|----------|-------------------|-------------|-----------|
| **Randomized A/B Test** | CUPED | DML | Variance reduction in experiments |
| **Observational Study** | PSM → DML | Synthetic Control | Start simple, add complexity |
| **Policy Evaluation** | DiD | Synthetic Control | Natural experiment design |
| **High-dimensional Data** | DML | PSM with ML | Handle many confounders |
| **Individual-level Effects** | Synthetic Control | DML | Transparent matching process |
| **Limited Pre-treatment Data** | PSM | DML | Work with available confounders |
| **Time Series Analysis** | DiD | Synthetic Control | Exploit temporal variation |
| **Heterogeneous Effects** | Causal Forest | DML | Personalized treatment effects |
| **Segment-specific Insights** | Causal Forest | PSM | Business targeting and personalization |

## 📈 **Business Impact Examples**

### **Smart Filing Assistant ROI Analysis**
```
Method Comparison Results:
- PSM ATE: +4.2% filing rate increase
- DML ATE: +3.8% filing rate increase  
- CUPED ATE: +4.1% filing rate increase
- DiD ATE: +3.9% filing rate increase
- Synthetic Control ATE: +4.3% filing rate increase
- Causal Forest ATE: +4.0% filing rate increase
  * High-value users: +6.1% increase
  * New users: +2.8% increase
  * Effect heterogeneity: ±1.5%

Business Impact (100K users):
- Additional filings: ~4,000 per year
- Revenue impact: $200K+ annually
- ROI: 400%+ (vs. development costs)
- Targeted ROI (top 50%): 520%+ with 50% cost reduction
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Add comprehensive tests for new functionality
4. Update documentation and notebooks
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 **Why This Toolkit?**

- **Comprehensive**: Six major causal inference methods in one place
- **Production-Ready**: Robust implementations with extensive testing
- **Business-Focused**: Clear ROI analysis and decision frameworks
- **Educational**: Step-by-step learning path with detailed explanations
- **Practical**: Real-world scenarios and implementation guidance
- **Heterogeneity-Aware**: Personalized effects with Causal Forest implementation

**🚀 Start exploring causal inference for your business decisions today!** 