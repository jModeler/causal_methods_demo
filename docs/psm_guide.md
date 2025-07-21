# Propensity Score Matching (PSM) Guide

## 📖 Overview

Propensity Score Matching (PSM) is a statistical technique used to estimate causal effects in observational data by reducing selection bias. This guide provides a comprehensive explanation of our PSM implementation and how to use it effectively.

## 🎯 What is Propensity Score Matching?

PSM addresses the fundamental challenge in causal inference: **how do we estimate what would have happened to treated units if they hadn't received treatment?** 

### The Problem
- In observational data, treatment assignment is not random
- Units that receive treatment may systematically differ from those that don't
- Simple comparisons of outcomes between treated and control groups can be biased

### The Solution
PSM creates a "pseudo-randomized" experiment by:
1. **Estimating propensity scores**: Probability of receiving treatment given observed characteristics
2. **Matching**: Finding control units with similar propensity scores to treated units
3. **Balancing**: Ensuring treated and control groups have similar distributions of confounders
4. **Estimating effects**: Comparing outcomes between matched groups

## 🔬 When to Use PSM

### ✅ **Good Use Cases**
- **Binary treatments**: Clear treatment/control distinction
- **Rich covariate data**: Many observed potential confounders
- **Cross-sectional analysis**: When temporal variation isn't available
- **Selection on observables**: When key confounders are measured

### ❌ **Not Ideal For**
- **Unobserved confounding**: PSM can't control for unmeasured variables
- **Continuous treatments**: Better methods exist for non-binary treatments
- **Poor overlap**: When treated and control groups are too different
- **Time-varying confounders**: Temporal methods like DiD may be better

## 🛠️ Implementation Overview

### Core Class: `PropensityScoreMatching`

```python
from src.causal_methods.psm import PropensityScoreMatching

# Initialize with your data
psm = PropensityScoreMatching(your_dataframe)
```

### Key Methods

1. **`estimate_propensity_scores()`**: Fit logistic regression model
2. **`perform_matching()`**: Match treated and control units
3. **`assess_balance()`**: Check covariate balance before/after matching
4. **`estimate_treatment_effects()`**: Calculate treatment effects on matched sample

## 📊 Step-by-Step Workflow

### Step 1: Propensity Score Estimation

```python
# Define covariates (potential confounders)
covariates = [
    'age', 'tech_savviness', 'income_bracket', 
    'device_type', 'user_type', 'region', 
    'filed_2023', 'early_login_2024'
]

# Estimate propensity scores
ps_results = psm.estimate_propensity_scores(
    treatment_col='used_smart_assistant',
    covariates=covariates,
    include_interactions=False  # Set to True for interaction terms
)

print(f"Model AUC: {ps_results['auc_score']:.3f}")
print(f"Propensity score range: {ps_results['propensity_score_range']}")
```

**Key Outputs:**
- `auc_score`: Model performance (higher = better prediction)
- `propensity_score_range`: Distribution of estimated scores
- `common_support`: Overlap between treatment groups
- `feature_importance`: Which variables predict treatment

### Step 2: Matching

```python
# Perform nearest neighbor matching
matching_results = psm.perform_matching(
    method='nearest_neighbor',  # or 'caliper'
    caliper=0.1,                # Maximum distance for matches
    replacement=False,          # Whether to allow reusing controls
    ratio=1                     # Controls per treated unit
)

print(f"Matching rate: {matching_results['matching_rate']:.1%}")
print(f"Matched sample size: {len(psm.matched_data)}")
```

**Matching Methods:**
- **Nearest Neighbor**: Find closest control for each treated unit
- **Caliper**: Only match within specified distance
- **With/Without Replacement**: Reuse controls or not

**Key Parameters:**
- `caliper`: Smaller = better matches but fewer observations
- `replacement`: True = more matches but potential bias
- `ratio`: Higher = more statistical power but worse matches

### Step 3: Balance Assessment

```python
# Check if matching improved balance
balance_results = psm.assess_balance(
    covariates=covariates,
    treatment_col='used_smart_assistant'
)

# Visualize balance improvement
fig = psm.plot_balance_assessment()
```

**Balance Metrics:**
- **Standardized Mean Difference (SMD)**: |Treated Mean - Control Mean| / Pooled SD
- **Guidelines**: SMD < 0.1 = excellent, SMD < 0.25 = acceptable
- **P-values**: Should be non-significant after good matching

### Step 4: Treatment Effect Estimation

```python
# Estimate effects on matched sample
effects = psm.estimate_treatment_effects(
    outcome_cols=['filed_2024', 'satisfaction_2024'],
    treatment_col='used_smart_assistant'
)

for outcome, result in effects.items():
    print(f"{outcome}:")
    print(f"  ATE: {result['ate']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  P-value: {result['p_value']:.4f}")
```

## 📈 Interpreting Results

### Propensity Score Model
- **AUC > 0.7**: Good predictive model
- **AUC < 0.6**: Weak selection model (good for causal inference!)
- **Check feature importance**: Which variables drive treatment assignment

### Matching Quality
- **Matching rate > 80%**: Excellent overlap
- **Matching rate 50-80%**: Good overlap
- **Matching rate < 50%**: Poor overlap, consider alternatives

### Balance Assessment
- **SMD reduction**: Should decrease for most variables after matching
- **P-values**: Should become non-significant for balanced variables
- **Visual check**: Distributions should look similar after matching

### Treatment Effects
- **Statistical significance**: P-value < 0.05 for significant effects
- **Practical significance**: Consider effect size magnitude
- **Confidence intervals**: Precision of estimates

## ⚠️ Common Issues and Solutions

### Issue 1: Poor Overlap
**Symptoms**: Low matching rates, extreme propensity scores
**Solutions**: 
- Reconsider treatment definition
- Include more relevant covariates
- Use caliper matching with appropriate threshold

### Issue 2: Imbalanced Covariates
**Symptoms**: High SMDs after matching
**Solutions**:
- Tighter calipers
- Include interaction terms
- Consider exact matching on key variables

### Issue 3: NaN P-values
**Symptoms**: Invalid statistical test results
**Solutions**:
- Our implementation automatically handles this with proportion tests for binary outcomes
- Check data quality and sample sizes

### Issue 4: Unstable Results
**Symptoms**: Results change significantly with small parameter changes
**Solutions**:
- Increase sample size
- Sensitivity analysis with multiple specifications
- Bootstrap confidence intervals

## 🎨 Visualization Tools

### Propensity Score Distributions
```python
# Compare distributions before/after matching
fig = psm.plot_propensity_distribution()
```

### Balance Assessment
```python
# Visualize covariate balance improvement
fig = psm.plot_balance_assessment()
```

### Treatment Effects
```python
# Plot effect estimates with confidence intervals
fig = psm.plot_treatment_effects()
```

## 🔍 Advanced Features

### Custom Covariate Selection
```python
# Automated selection based on correlation
high_corr_vars = df.corr()['used_smart_assistant'].abs().sort_values(ascending=False)
top_covariates = high_corr_vars.head(10).index.tolist()
```

### Sensitivity Analysis
```python
# Test different calipers
calipers = [0.05, 0.1, 0.15, 0.2]
results = []

for cal in calipers:
    psm_test = PropensityScoreMatching(df)
    psm_test.estimate_propensity_scores(covariates=covariates)
    psm_test.perform_matching(caliper=cal)
    if psm_test.matched_data is not None:
        effects = psm_test.estimate_treatment_effects(outcome_cols='filed_2024')
        results.append({
            'caliper': cal,
            'ate': effects['filed_2024']['ate'],
            'matching_rate': len(psm_test.matched_data) / len(df)
        })
```

### Interaction Terms
```python
# Include interaction effects in propensity model
ps_results = psm.estimate_propensity_scores(
    covariates=covariates,
    include_interactions=True  # Adds pairwise interactions
)
```

## 📚 Theoretical Background

### Assumptions
1. **Unconfoundedness**: No unobserved confounders affect both treatment and outcome
2. **Common Support**: Overlap in propensity score distributions
3. **SUTVA**: Stable Unit Treatment Value Assumption (no interference)

### Advantages
- **Intuitive**: Easy to understand and explain
- **Flexible**: Works with any outcome type
- **Transparent**: Clear assessment of balance and match quality
- **Non-parametric**: No assumptions about outcome model

### Limitations
- **Selection on observables**: Can't handle unobserved confounding
- **Efficiency**: May lose observations in matching
- **Complexity**: Requires careful tuning and assessment

## 🎯 Best Practices

### Before Analysis
1. **Understand your data**: Explore treatment assignment patterns
2. **Domain knowledge**: Use subject expertise to select covariates
3. **Pre-specify analysis**: Define outcomes and methods beforehand

### During Analysis
1. **Check overlap**: Ensure common support exists
2. **Balance assessment**: Always check before proceeding
3. **Sensitivity analysis**: Test robustness to specification choices
4. **Visualize**: Use plots to understand your data and results

### After Analysis
1. **Interpret carefully**: Consider practical vs statistical significance
2. **Acknowledge limitations**: Discuss potential confounders
3. **Triangulate**: Compare with other methods if possible
4. **Document decisions**: Record analysis choices and rationale

## 🚀 Next Steps

After mastering PSM:
1. **Compare methods**: Try DiD for temporal variation
2. **Advanced techniques**: Explore double machine learning
3. **Sensitivity analysis**: Test for unobserved confounding
4. **Integration**: Combine multiple causal methods

## 📖 Further Reading

- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects.
- Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding.
- Stuart, E. A. (2010). Matching methods for causal inference: A review and a look forward.

---

This guide provides the foundation for effective PSM analysis. For hands-on practice, work through the [PSM demonstration notebook](../notebooks/02_psm_tax.ipynb)! 