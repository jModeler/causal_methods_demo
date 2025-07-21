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

## 📊 Implemented Methods

### 1. **Propensity Score Matching (PSM)**
- **Purpose**: Matches treated and control units based on propensity scores
- **Implementation**: `src/causal_methods/psm.py`
- **Notebook**: `notebooks/02_psm_tax.ipynb`
- **Key Features**:
  - Multiple matching algorithms (nearest neighbor, caliper matching)
  - Automated covariate balance assessment  
  - Statistical significance testing with proper binary outcome handling
  - Rich visualizations (propensity distributions, balance plots, effect plots)
  - Robust error handling and edge case management

### 2. **Double Machine Learning (DML)** ✨ *Enhanced with Information Criteria!*
- **Purpose**: Combines ML prediction with causal inference using cross-fitting
- **Implementation**: `src/causal_methods/dml.py`
- **Notebook**: `notebooks/03_dml_tax.ipynb`
- **Key Features**:
  - **Cross-fitting with K-fold splits** to avoid overfitting bias
  - **Multiple ML algorithms**: Random Forest, Gradient Boosting, Linear/Logistic Regression, Ridge
  - **Information Criteria Integration**: AIC and BIC for principled model selection
  - **Robust Model Performance**: Handles both continuous and binary outcomes
  - **Comprehensive Diagnostics**: Model comparison, residual analysis, treatment effect visualization
  - **Statistical Inference**: Confidence intervals, p-values, influence function-based standard errors
  - **Edge Case Handling**: Single-class predictions, missing predict_proba methods, perfect separation
  - **Multiple Outcomes**: Estimate effects on several outcome variables simultaneously

### 3. **CUPED (Controlled-experiment Using Pre-Experiment Data)**
- **Purpose**: Variance reduction technique for randomized experiments
- **Implementation**: `src/causal_methods/cuped.py`
- **Notebook**: `notebooks/04_cuped_tax.ipynb`
- **Key Features**:
  - Uses pre-experiment covariates to increase statistical power
  - Preserves unbiasedness while reducing confidence intervals
  - Optimal adjustment coefficient estimation (θ)
  - Multiple regression methods (OLS, Ridge, Lasso)
  - Covariate balance checking for randomized experiments
  - Substantial variance reduction and power improvements

### 4. **Difference-in-Differences (DiD)**
- **Purpose**: For before/after treatment analysis with panel data
- **Implementation**: `src/causal_methods/did.py`
- **Notebook**: `notebooks/01_did_tax.ipynb`
- **Key Features**:
  - Controls for time-invariant confounders
  - Parallel trends assumption testing
  - Robust standard errors with clustering
  - Panel data preparation and visualization

## 🚀 Quick Start

### Double Machine Learning with Information Criteria
```python
from src.causal_methods.dml import DoubleMachineLearning

# Initialize DML
dml = DoubleMachineLearning(data, random_state=42)

# Estimate treatment effects with model comparison
results = dml.estimate_treatment_effects(
    outcome_col='revenue',
    treatment_col='used_feature',
    covariates=['age', 'income', 'past_purchases'],
    outcome_model='random_forest',
    treatment_model='gradient_boosting',
    n_folds=5
)

print(f"ATE: {results['ate']:.4f}")
print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
print(f"P-value: {results['p_value']:.4f}")

# Compare multiple ML models with information criteria
comparison = dml.compare_models(
    outcome_col='revenue',
    treatment_col='used_feature',
    covariates=['age', 'income', 'past_purchases']
)

# Find best models by information criteria
best_aic = comparison.loc[comparison['aic'].idxmin()]
best_bic = comparison.loc[comparison['bic'].idxmin()]

print(f"Best AIC model: {best_aic['outcome_model']} + {best_aic['treatment_model']}")
print(f"Best BIC model: {best_bic['outcome_model']} + {best_bic['treatment_model']}")

# Generate comprehensive report
report = dml.generate_summary_report('revenue')
print(report)
```

### CUPED for Randomized Experiments
```python
from src.causal_methods.cuped import CUPED

# Initialize CUPED
cuped = CUPED(data, random_state=42)

# Estimate treatment effects with variance reduction
results = cuped.estimate_treatment_effects(
    outcome_col='conversion_rate',
    treatment_col='treatment_group',
    covariate_cols=['baseline_conversion', 'user_engagement', 'past_activity']
)

print(f"Variance reduction: {results['summary']['variance_reduction']:.1%}")
print(f"Power improvement: {results['summary']['power_improvement']:.1f}×")
print(f"SE reduction: {results['summary']['se_reduction']:.1%}")

# Visualize CUPED impact
fig = cuped.plot_cuped_comparison('conversion_rate', 'treatment_group')
```

### Propensity Score Matching
```python
from src.causal_methods.psm import PropensityScoreMatching

# Initialize PSM
psm = PropensityScoreMatching(data)

# Estimate propensity scores
ps_results = psm.estimate_propensity_scores(
    covariates=['age', 'income', 'education']
)

# Perform matching with caliper
matching_results = psm.perform_matching(
    method='nearest_neighbor', 
    caliper=0.1
)

# Estimate treatment effects
effects = psm.estimate_treatment_effects(outcome_cols='outcome')

print(f"ATE: {effects['outcome']['ate']:.4f}")
print(f"Matching rate: {matching_results['matching_rate']:.1%}")
```

## 🔬 Information Criteria in Model Selection

Our DML implementation features advanced model selection using **information criteria**:

### Why Information Criteria vs R²?

- **Account for Model Complexity**: Penalize overfitting appropriately
- **Fair Model Comparison**: Compare different algorithms on equal footing  
- **Statistical Rigor**: Standard approach in econometrics and statistics
- **Better Generalization**: Choose models that predict well out-of-sample

### Available Metrics

- **AIC (Akaike Information Criterion)**: Balances fit vs complexity, favors prediction
- **BIC (Bayesian Information Criterion)**: Stronger complexity penalty, favors parsimony  
- **Log-Likelihood**: Goodness of fit measure
- **Parameter Count**: Effective number of model parameters

### Model Selection Workflow

```python
# Compare models with information criteria
comparison = dml.compare_models(...)

# Best model selection
best_aic_model = comparison.loc[comparison['aic'].idxmin()]
best_bic_model = comparison.loc[comparison['bic'].idxmin()]

# Information criteria statistics
print(f"AIC Range: [{comparison['aic'].min():.2f}, {comparison['aic'].max():.2f}]")
print(f"BIC Range: [{comparison['bic'].min():.2f}, {comparison['bic'].max():.2f}]")
```

## 📈 Method Comparison Guide

| Method | Best Use Case | Key Strength | Limitation |
|--------|---------------|--------------|------------|
| **CUPED** | Randomized experiments with pre-data | High precision, preserves randomization | Requires good pre-experiment predictors |
| **DML** | High-dimensional observational data | Robust to model misspecification | Complex, requires cross-fitting |
| **PSM** | Observational data with good covariates | Controls for observed confounders | Can't control unobserved confounders |
| **DiD** | Before/after natural experiments | Controls for time-invariant confounders | Needs parallel trends assumption |

## 🧪 Testing & Quality Assurance

### Comprehensive Test Suite
- **130+ unit tests** covering all methods and edge cases
- **Integration tests** comparing methods on same datasets  
- **Edge case handling**: Small samples, perfect separation, missing data
- **Continuous integration** with pytest and quality checks

### Code Quality Standards
- **Linting**: Automated code formatting and style checks with ruff
- **Type hints**: Full type annotation for better IDE support
- **Documentation**: Comprehensive docstrings and examples
- **Error handling**: Robust error messages and graceful failures

### Test Coverage Areas
- ✅ **DML**: Treatment effects, multiple outcomes, model comparison, information criteria, cross-fitting
- ✅ **CUPED**: Variance reduction, covariate balance, binary/continuous outcomes  
- ✅ **PSM**: Propensity scores, matching algorithms, balance assessment
- ✅ **DiD**: Panel data, parallel trends, clustering 