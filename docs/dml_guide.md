# Double Machine Learning (DML) Guide

## 🎯 Overview

Double Machine Learning (DML) is a cutting-edge causal inference method that combines the flexibility of machine learning with the rigor of causal inference. Our implementation features **information criteria-based model selection** for superior performance and reliability.

## 🔬 Methodology

### Core Concept

DML estimates causal effects by:

1. **First Stage**: Use ML to predict outcomes and treatments separately
2. **Second Stage**: Estimate causal effects using residualized predictions
3. **Cross-fitting**: Split data to avoid overfitting bias
4. **Orthogonal Moments**: Debiased estimation robust to model misspecification

### Mathematical Foundation

For outcome Y, treatment D, and covariates X:

```
Y = g(X) + θ·D + ε₁    (Outcome equation)
D = m(X) + ε₂          (Treatment equation)
```

DML estimates θ (the treatment effect) using:
```
θ̂ = E[(Y - ĝ(X))(D - m̂(X))] / E[(D - m̂(X))²]
```

## 🚀 Key Features

### ✨ Information Criteria Integration

Our DML implementation goes beyond traditional R² metrics by incorporating:

- **AIC (Akaike Information Criterion)**: Balances model fit and complexity
- **BIC (Bayesian Information Criterion)**: Stronger penalty for model complexity
- **Log-Likelihood**: Direct measure of model fit quality
- **Parameter Count**: Tracks effective model complexity

### 🔧 Advanced Capabilities

- **Multiple ML Algorithms**: Random Forest, Gradient Boosting, Linear/Logistic Regression, Ridge
- **Cross-fitting**: K-fold sample splitting prevents overfitting
- **Robust Inference**: Influence function-based standard errors
- **Multiple Outcomes**: Simultaneous estimation for several outcome variables
- **Model Comparison**: Systematic comparison across algorithm combinations
- **Comprehensive Diagnostics**: Residual plots, effect visualization, performance metrics

## 📊 Usage Examples

### Basic Treatment Effect Estimation

```python
from src.causal_methods.dml import DoubleMachineLearning

# Initialize DML
dml = DoubleMachineLearning(data, random_state=42)

# Estimate treatment effects
results = dml.estimate_treatment_effects(
    outcome_col='revenue',
    treatment_col='used_feature',
    covariates=['age', 'income', 'past_purchases'],
    outcome_model='random_forest',
    treatment_model='logistic_regression',
    n_folds=5,
    scale_features=True
)

# View results
print(f"Average Treatment Effect: {results['ate']:.4f}")
print(f"Standard Error: {results['se']:.4f}")
print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
print(f"P-value: {results['p_value']:.6f}")
```

### Model Comparison with Information Criteria

```python
# Compare all model combinations
comparison = dml.compare_models(
    outcome_col='revenue',
    treatment_col='used_feature',
    covariates=['age', 'income', 'past_purchases'],
    n_folds=3
)

# Display results sorted by AIC
print("Model Comparison (sorted by AIC):")
print(comparison.sort_values('aic')[['outcome_model', 'treatment_model', 'ate', 'aic', 'bic']].round(4))

# Find best models
best_aic = comparison.loc[comparison['aic'].idxmin()]
best_bic = comparison.loc[comparison['bic'].idxmin()]

print(f"\nBest AIC Model: {best_aic['outcome_model']} + {best_aic['treatment_model']}")
print(f"  ATE: {best_aic['ate']:.4f}, AIC: {best_aic['aic']:.2f}")

print(f"\nBest BIC Model: {best_bic['outcome_model']} + {best_bic['treatment_model']}")
print(f"  ATE: {best_bic['ate']:.4f}, BIC: {best_bic['bic']:.2f}")
```

### Multiple Outcomes Analysis

```python
# Estimate effects on multiple outcomes
outcomes = ['revenue', 'engagement', 'retention']

for outcome in outcomes:
    results = dml.estimate_treatment_effects(
        outcome_col=outcome,
        treatment_col='used_feature',
        covariates=['age', 'income', 'past_purchases']
    )
    
    print(f"\n{outcome.upper()} Results:")
    print(f"  ATE: {results['ate']:.4f}")
    print(f"  P-value: {results['p_value']:.6f}")
    print(f"  Significant: {'Yes' if results['p_value'] < 0.05 else 'No'}")
```

### Visualization and Diagnostics

```python
# Plot residuals for model diagnostics
fig = dml.plot_residuals('revenue', figsize=(12, 5))
plt.show()

# Plot treatment effects
fig = dml.plot_treatment_effects(figsize=(10, 6))
plt.show()

# Generate comprehensive report
report = dml.generate_summary_report('revenue')
print(report)
```

## 🎛️ Configuration Options

### Available ML Models

| Model | Type | Use Case | Parameters |
|-------|------|----------|------------|
| `'random_forest'` | Tree-based ensemble | General purpose, handles non-linearity | n_estimators=100 |
| `'gradient_boosting'` | Tree-based ensemble | High performance, good for complex patterns | n_estimators=100 |
| `'linear_regression'` | Linear | Interpretable, fast | - |
| `'logistic_regression'` | Linear | Binary outcomes/treatments | max_iter=1000 |
| `'ridge'` | Regularized linear | High-dimensional data | alpha=1.0 |

### Cross-fitting Options

```python
# Different cross-fitting configurations
results = dml.estimate_treatment_effects(
    outcome_col='revenue',
    treatment_col='used_feature',
    covariates=covariates,
    n_folds=5,                    # Number of folds (2-10 recommended)
    scale_features=True,          # Standardize features
    random_state=42              # Reproducibility
)
```

### Feature Scaling

- **Recommended**: `scale_features=True` for most ML algorithms
- **Exception**: Tree-based methods (Random Forest, Gradient Boosting) don't require scaling
- **Automatic**: Applied only to numeric features

## 📈 Information Criteria Deep Dive

### Why Information Criteria?

Traditional metrics like R² can be misleading because they don't account for model complexity:

- **Overfitting Risk**: High R² might indicate overfitting
- **Model Complexity**: Simpler models often generalize better
- **Fair Comparison**: IC enables comparison across different algorithm types

### AIC vs BIC

| Criterion | Formula | Penalty | Best For |
|-----------|---------|---------|----------|
| **AIC** | 2k - 2ln(L) | Moderate | Prediction focus |
| **BIC** | k·ln(n) - 2ln(L) | Strong | Model selection |

Where:
- k = number of parameters
- n = sample size  
- L = likelihood

### Interpretation Guidelines

```python
# Model selection based on information criteria
comparison = dml.compare_models(...)

# Lower values are better
best_models = comparison.nsmallest(3, 'aic')  # Top 3 by AIC
print("Top 3 models by AIC:")
print(best_models[['outcome_model', 'treatment_model', 'aic', 'bic']])

# Check for substantial differences
aic_range = comparison['aic'].max() - comparison['aic'].min()
if aic_range > 10:
    print("⚠️ Large AIC differences suggest some models much better than others")
else:
    print("✅ Models perform similarly - other factors may guide selection")
```

## 🔍 Model Diagnostics

### Residual Analysis

Good DML models should have:
- **Centered residuals**: Mean ≈ 0
- **No patterns**: Random scatter in residual plots
- **Similar variances**: Homoscedasticity across treatment groups

```python
# Automated residual plotting
fig = dml.plot_residuals('outcome_col')

# Look for:
# - Centered distributions
# - No clear patterns
# - Similar spread across groups
```

### Performance Metrics

Monitor these key metrics for model quality:

```python
# Access detailed performance
results = dml.estimate_treatment_effects(...)

# Check fold-level performance
for fold_data in results['fold_performance']:
    outcome_perf = fold_data['outcome_performance']
    treatment_perf = fold_data['treatment_performance']
    
    print(f"Fold {fold_data['fold']}:")
    print(f"  Outcome AIC: {outcome_perf['aic']:.2f}")
    print(f"  Treatment AIC: {treatment_perf['aic']:.2f}")
```

## ⚠️ Common Pitfalls & Solutions

### 1. Poor Model Performance

**Problem**: High AIC/BIC values, poor predictive metrics
```python
# Check if features are predictive
correlations = data[covariates].corrwith(data[outcome_col])
print("Feature correlations with outcome:")
print(correlations.abs().sort_values(ascending=False))
```

**Solutions**:
- Add more relevant features
- Try different ML algorithms
- Feature engineering (interactions, polynomials)
- Check for data quality issues

### 2. Perfect Separation

**Problem**: Treatment perfectly predicts outcome
```python
# Check for perfect separation
cross_tab = pd.crosstab(data[treatment_col], data[outcome_col])
print(cross_tab)
```

**Solutions**:
- Add noise/regularization
- Use Ridge regression
- Collect more diverse data

### 3. Insufficient Cross-fitting

**Problem**: Overfitting bias despite DML
```python
# Use more folds for better debiasing
results = dml.estimate_treatment_effects(
    ...,
    n_folds=10  # Increase from default 5
)
```

### 4. Scale Sensitivity

**Problem**: Inconsistent results across runs
```python
# Always scale features for consistency
results = dml.estimate_treatment_effects(
    ...,
    scale_features=True,
    random_state=42  # Fix random state
)
```

## 🎯 Best Practices

### 1. Model Selection Strategy

1. **Start broad**: Compare all algorithm combinations
2. **Focus on IC**: Prioritize AIC/BIC over traditional metrics
3. **Check robustness**: Verify results across similar-performing models
4. **Consider context**: Balance performance with interpretability needs

### 2. Sample Size Guidelines

- **Minimum**: 500+ observations for reliable DML
- **Recommended**: 1000+ for stable cross-fitting
- **High-dimensional**: 10+ observations per feature

### 3. Feature Engineering

```python
# Good practice: Create interaction features
data['age_income'] = data['age'] * data['income']
data['age_squared'] = data['age'] ** 2

# Include in covariates
covariates = ['age', 'income', 'age_income', 'age_squared', ...]
```

### 4. Validation Workflow

```python
# 1. Check data quality
print("Missing values:", data.isnull().sum())
print("Treatment balance:", data[treatment_col].value_counts(normalize=True))

# 2. Run model comparison
comparison = dml.compare_models(...)

# 3. Validate best model
best_model = comparison.loc[comparison['aic'].idxmin()]
results = dml.estimate_treatment_effects(
    outcome_model=best_model['outcome_model'],
    treatment_model=best_model['treatment_model'],
    ...
)

# 4. Check diagnostics
fig = dml.plot_residuals(outcome_col)
report = dml.generate_summary_report(outcome_col)
```

## 📚 Technical References

### Academic Foundation

- **Chernozhukov et al. (2018)**: "Double/debiased machine learning for treatment and structural parameters"
- **Belloni et al. (2017)**: "Program evaluation and causal inference with high-dimensional data"
- **Akaike (1974)**: "A new look at the statistical model identification"
- **Schwarz (1978)**: "Estimating the dimension of a model"

### Implementation Details

Our DML implementation includes several enhancements:

- **Robust standard errors**: Influence function-based inference
- **Flexible ML integration**: Easy to extend with new algorithms
- **Information criteria**: Principled model selection beyond R²
- **Edge case handling**: Graceful failures for challenging data

### Performance Characteristics

- **Time complexity**: O(n log n) for tree-based methods
- **Memory usage**: Linear in sample size and features
- **Convergence**: Guaranteed for convex problems (linear methods)
- **Scalability**: Efficient for datasets up to 100K+ observations

## 🤝 Integration Examples

### With Pandas

```python
import pandas as pd

# Clean data integration
df = pd.read_csv('your_data.csv')
df = df.dropna()  # Handle missing values

# Automatic column detection
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# DML works best with numeric features
dml = DoubleMachineLearning(df)
results = dml.estimate_treatment_effects(
    outcome_col='your_outcome',
    treatment_col='your_treatment',
    covariates=numeric_cols.tolist()
)
```

### With Scikit-learn

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Advanced preprocessing
scaler = StandardScaler()
encoder = LabelEncoder()

# DML handles this internally, but you can pre-process
df_processed = df.copy()
df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])

dml = DoubleMachineLearning(df_processed)
```

## 🎓 Learning Path

### Beginner
1. Start with `notebooks/03_dml_tax.ipynb`
2. Run basic treatment effect estimation
3. Understand cross-fitting concept

### Intermediate  
1. Experiment with different ML algorithms
2. Learn information criteria interpretation
3. Practice model comparison workflow

### Advanced
1. Implement custom ML algorithms
2. Explore influence function theory
3. Contribute to method enhancements

---

**📝 Note**: This guide covers our enhanced DML implementation with information criteria. For theoretical background, consult the academic references. For implementation details, see the source code in `src/causal_methods/dml.py`. 