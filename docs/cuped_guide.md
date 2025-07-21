# CUPED (Controlled-experiment Using Pre-Experiment Data) Guide

## 🎯 Overview

CUPED is a variance reduction technique specifically designed for **randomized controlled experiments** (A/B tests). It leverages pre-experiment data to dramatically improve the precision of treatment effect estimates while preserving the unbiasedness that makes randomized experiments so valuable.

## 🔬 Methodology

### Core Concept

CUPED works by applying a simple but powerful adjustment to outcome variables:

```
Y_cuped = Y - θ × (X - E[X])
```

Where:
- **Y** = Original outcome variable
- **X** = Pre-experiment covariate(s) 
- **θ** = Optimal adjustment coefficient
- **E[X]** = Expected value (mean) of the covariate

This adjustment **reduces variance** without introducing bias, leading to:
- Narrower confidence intervals
- Higher statistical power
- Faster time to significance
- More cost-effective experiments

### Mathematical Foundation

The optimal adjustment coefficient θ is estimated via regression:

```
Y = α + θ·X + ε
```

In randomized experiments, this coefficient represents the **correlation** between pre-experiment and post-experiment measures, not a causal relationship.

### Why CUPED Works

1. **Removes correlated noise**: Pre-experiment data explains variance in outcomes
2. **Preserves randomization**: Treatment assignment remains random
3. **Maintains unbiasedness**: E[Y_cuped | Treatment] = E[Y | Treatment]
4. **Reduces variance**: Var(Y_cuped) ≤ Var(Y)

## 🚀 Key Features

### ✨ Variance Reduction
- **Substantial precision gains**: Often 20-50% variance reduction
- **Statistical power improvement**: 1.5-2× power increase common
- **Faster experiments**: Reach significance with smaller samples
- **Cost savings**: More information from existing data

### 🔧 Robust Implementation
- **Multiple adjustment methods**: OLS, Ridge, Lasso regression
- **Automatic coefficient estimation**: Optimal θ calculation
- **Covariate balance checking**: Validates randomization assumptions
- **Missing value handling**: Graceful degradation with incomplete data
- **Edge case protection**: Handles perfect correlation, small samples

### 📊 Comprehensive Analysis
- **Treatment effect comparison**: Original vs CUPED-adjusted
- **Diagnostic visualizations**: Distribution plots, confidence intervals
- **Business impact metrics**: Effect sizes, practical significance
- **Statistical reporting**: P-values, confidence intervals, power analysis

## 📊 Usage Examples

### Basic CUPED Analysis

```python
from src.causal_methods.cuped import CUPED

# Initialize CUPED analyzer
cuped = CUPED(data, random_state=42)

# Estimate treatment effects with variance reduction
results = cuped.estimate_treatment_effects(
    outcome_col='conversion_rate',
    treatment_col='treatment_group', 
    covariate_cols=['baseline_conversion', 'past_engagement'],
    confidence_level=0.95
)

# View key results
print(f"Original ATE: {results['original']['ate']:.4f}")
print(f"CUPED ATE: {results['cuped']['ate']:.4f}")
print(f"Variance reduction: {results['summary']['variance_reduction']:.1%}")
print(f"Power improvement: {results['summary']['power_improvement']:.1f}×")
```

### Advanced CUPED Workflow

```python
# Step 1: Estimate adjustment coefficients
adjustment_info = cuped.estimate_cuped_adjustment(
    outcome_col='conversion_rate',
    covariate_cols=['baseline_conversion', 'user_engagement', 'past_revenue'],
    treatment_col='treatment_group',
    method='ridge'  # Use Ridge regression for stability
)

print(f"Adjustment R²: {adjustment_info['r2']:.3f}")
print(f"Expected variance reduction: {adjustment_info['variance_reduction']:.1%}")

# Step 2: Check covariate balance (crucial for randomized experiments)
if 'balance_check' in adjustment_info:
    for covar, stats in adjustment_info['balance_check'].items():
        balance_status = "✅ Balanced" if stats['balanced'] else "⚠️ Imbalanced"
        print(f"{covar}: {balance_status} (std diff: {stats['std_diff']:.3f})")

# Step 3: Apply CUPED adjustment
cuped_outcome = cuped.apply_cuped_adjustment('conversion_rate')

# Step 4: Comprehensive visualization
fig = cuped.plot_cuped_comparison(
    outcome_col='conversion_rate',
    treatment_col='treatment_group',
    figsize=(16, 12)
)
plt.show()

# Step 5: Generate business report
report = cuped.generate_summary_report('conversion_rate')
print(report)
```

### Multiple Covariates Analysis

```python
# Compare different covariate combinations
covariate_sets = [
    ['baseline_conversion'],
    ['baseline_conversion', 'user_engagement'], 
    ['baseline_conversion', 'user_engagement', 'past_revenue'],
    ['baseline_conversion', 'user_engagement', 'past_revenue', 'session_count']
]

results_comparison = {}

for i, covariates in enumerate(covariate_sets, 1):
    try:
        results = cuped.estimate_treatment_effects(
            outcome_col='conversion_rate',
            treatment_col='treatment_group',
            covariate_cols=covariates
        )
        
        results_comparison[f'Set_{i}'] = {
            'covariates': covariates,
            'variance_reduction': results['summary']['variance_reduction'],
            'power_improvement': results['summary']['power_improvement'],
            'se_reduction': results['summary']['se_reduction'],
            'p_value': results['cuped']['p_value']
        }
    except Exception as e:
        print(f"Set {i} failed: {e}")

# Display comparison
import pandas as pd
comparison_df = pd.DataFrame(results_comparison).T
print("Covariate Set Comparison:")
print(comparison_df.round(3))
```

### Time Series Pre-Experiment Data

```python
# Using multiple time periods as covariates
time_covariates = [
    'conversion_week_1', 'conversion_week_2', 'conversion_week_3', 'conversion_week_4'
]

# Estimate with historical data
results = cuped.estimate_treatment_effects(
    outcome_col='conversion_rate_current',
    treatment_col='treatment_group',
    covariate_cols=time_covariates
)

print(f"Using 4 weeks of historical data:")
print(f"Variance reduction: {results['summary']['variance_reduction']:.1%}")
print(f"Statistical power increase: {results['summary']['power_improvement']:.1f}×")
```

## 🎛️ Configuration Options

### Adjustment Methods

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| `'ols'` | Standard case | Simple, interpretable | May overfit with many covariates |
| `'ridge'` | Many covariates | Handles multicollinearity | Requires tuning alpha parameter |
| `'lasso'` | Feature selection | Automatic variable selection | May be too aggressive |

```python
# Compare adjustment methods
methods = ['ols', 'ridge', 'lasso']
method_results = {}

for method in methods:
    adjustment_info = cuped.estimate_cuped_adjustment(
        outcome_col='conversion_rate',
        covariate_cols=['baseline_conversion', 'engagement', 'revenue'],
        method=method
    )
    method_results[method] = adjustment_info['r2']

print("Method Comparison (R²):")
for method, r2 in method_results.items():
    print(f"  {method.upper()}: {r2:.3f}")
```

### Confidence Levels

```python
# Different confidence levels for decision-making
confidence_levels = [0.90, 0.95, 0.99]

for conf_level in confidence_levels:
    results = cuped.estimate_treatment_effects(
        outcome_col='conversion_rate',
        treatment_col='treatment_group',
        covariate_cols=['baseline_conversion'],
        confidence_level=conf_level
    )
    
    ci_width = results['cuped']['ci_upper'] - results['cuped']['ci_lower']
    print(f"{conf_level*100}% CI width: {ci_width:.4f}")
```

## 📈 Choosing Effective Covariates

### What Makes a Good CUPED Covariate?

1. **Highly correlated with outcome**: |correlation| > 0.3 ideal
2. **Pre-experiment**: Measured before treatment assignment
3. **Balanced across groups**: Similar distributions in treatment/control
4. **Stable over time**: Reliable measurement
5. **Business-relevant**: Makes intuitive sense

### Covariate Selection Strategy

```python
# Analyze covariate potential
pre_experiment_vars = [
    'baseline_conversion', 'user_engagement', 'past_revenue', 
    'session_count', 'days_since_signup', 'feature_usage'
]

outcome_col = 'conversion_rate'

print("🔍 COVARIATE ANALYSIS:")
print("=" * 50)

correlations = []
for var in pre_experiment_vars:
    # Calculate correlation with outcome
    if data[outcome_col].dtype == 'bool' or data[outcome_col].nunique() == 2:
        corr = data[var].corr(data[outcome_col].astype(int))
    else:
        corr = data[var].corr(data[outcome_col])
    
    # Assess CUPED potential
    potential = 'High' if abs(corr) > 0.3 else 'Medium' if abs(corr) > 0.1 else 'Low'
    
    correlations.append({
        'Variable': var,
        'Correlation': corr,
        'CUPED_Potential': potential
    })
    
    print(f"  {var}: {corr:.3f} ({potential} potential)")

# Select best covariates
good_covariates = [c['Variable'] for c in correlations if abs(c['Correlation']) > 0.1]
print(f"\n✅ Recommended covariates: {good_covariates}")
```

### Avoiding Bad Covariates

❌ **Avoid these:**
- Post-treatment variables
- Outcome-dependent variables  
- Perfectly correlated variables (multicollinearity)
- Variables with extreme imbalance

```python
# Check for problematic covariates
def validate_covariates(data, covariates, treatment_col, outcome_col):
    issues = []
    
    for covar in covariates:
        # Check for perfect correlation
        corr_with_outcome = abs(data[covar].corr(data[outcome_col]))
        if corr_with_outcome > 0.99:
            issues.append(f"⚠️ {covar}: Perfect correlation with outcome")
        
        # Check for treatment-outcome dependence
        treated_mean = data[data[treatment_col] == 1][covar].mean()
        control_mean = data[data[treatment_col] == 0][covar].mean()
        std_diff = abs(treated_mean - control_mean) / data[covar].std()
        
        if std_diff > 0.2:
            issues.append(f"⚠️ {covar}: Large imbalance between groups")
    
    return issues

# Validate your covariates
issues = validate_covariates(data, good_covariates, 'treatment_group', 'conversion_rate')
for issue in issues:
    print(issue)
```

## 🔍 Interpreting CUPED Results

### Key Metrics to Monitor

```python
results = cuped.estimate_treatment_effects(...)

# 1. Variance Reduction
variance_reduction = results['summary']['variance_reduction']
if variance_reduction > 0.2:
    print("🎉 Excellent variance reduction (>20%)")
elif variance_reduction > 0.1:
    print("✅ Good variance reduction (10-20%)")
elif variance_reduction > 0:
    print("👍 Some variance reduction")
else:
    print("⚠️ No variance reduction - check covariates")

# 2. Statistical Power Improvement
power_improvement = results['summary']['power_improvement']
print(f"Statistical power improved by {power_improvement:.1f}×")

# 3. Significance Status
orig_sig = results['original']['p_value'] < 0.05
cuped_sig = results['cuped']['p_value'] < 0.05

if not orig_sig and cuped_sig:
    print("🚀 CUPED enabled significance detection!")
elif orig_sig and cuped_sig:
    print("✅ Consistent significance across methods")
else:
    print("📊 Results depend on method choice")
```

### Business Impact Translation

```python
# Translate CUPED results to business metrics
def business_impact_analysis(results, sample_size, outcome_type='conversion'):
    cuped_ate = results['cuped']['ate']
    orig_ate = results['original']['ate']
    
    # Confidence interval precision
    orig_ci_width = results['original']['ci_upper'] - results['original']['ci_lower']
    cuped_ci_width = results['cuped']['ci_upper'] - results['cuped']['ci_lower']
    precision_gain = (orig_ci_width - cuped_ci_width) / orig_ci_width
    
    # Sample size efficiency
    power_improvement = results['summary']['power_improvement']
    equivalent_sample_reduction = 1 - (1 / power_improvement**2)
    
    print(f"📊 BUSINESS IMPACT ANALYSIS")
    print(f"=" * 40)
    print(f"Treatment Effect: {cuped_ate:.1%} {outcome_type} increase")
    print(f"Confidence Interval: ±{cuped_ci_width/2:.1%}")
    print(f"Precision Improvement: {precision_gain:.1%}")
    print(f"Equivalent to {equivalent_sample_reduction:.1%} sample size reduction")
    
    if sample_size:
        potential_users_affected = sample_size * cuped_ate
        print(f"Potential Users Affected: {potential_users_affected:.0f}")

# Apply to your results
business_impact_analysis(results, sample_size=10000, outcome_type='conversion')
```

## ⚠️ Common Pitfalls & Solutions

### 1. Poor Covariate Selection

**Problem**: Little to no variance reduction
```python
# Diagnose weak covariates
adjustment_info = cuped.estimate_cuped_adjustment(...)
if adjustment_info['r2'] < 0.05:
    print("⚠️ Weak covariates - R² < 5%")
    print("💡 Try: More relevant pre-experiment data")
```

**Solutions**:
- Use more historically relevant variables
- Include user behavioral data
- Add demographic information
- Consider interaction terms

### 2. Covariate Imbalance

**Problem**: Randomization assumption violated
```python
# Check and fix imbalance
balance_check = adjustment_info['balance_check']
imbalanced_vars = [var for var, stats in balance_check.items() if not stats['balanced']]

if imbalanced_vars:
    print(f"⚠️ Imbalanced variables: {imbalanced_vars}")
    print("💡 Solutions:")
    print("  - Check randomization procedure")
    print("  - Use stratified randomization")
    print("  - Apply post-hoc balancing weights")
```

### 3. Overfitting with Many Covariates

**Problem**: R² looks good but results are unstable
```python
# Use regularized methods for many covariates
if len(covariate_cols) > 10:
    adjustment_info = cuped.estimate_cuped_adjustment(
        outcome_col='conversion_rate',
        covariate_cols=covariate_cols,
        method='ridge'  # Regularization prevents overfitting
    )
```

### 4. Missing Pre-Experiment Data

**Problem**: Incomplete historical data
```python
# Handle missing data gracefully
missing_pct = data[covariate_cols].isnull().mean()
problematic_vars = missing_pct[missing_pct > 0.1].index.tolist()

if problematic_vars:
    print(f"⚠️ High missing data: {problematic_vars}")
    print("💡 Options:")
    print("  - Use complete cases only") 
    print("  - Impute with care")
    print("  - Exclude high-missing variables")
```

## 🎯 Best Practices

### 1. Pre-Experiment Planning

```python
# BEFORE running your experiment, plan CUPED analysis
def plan_cuped_analysis(historical_data, outcome_col, potential_covariates):
    """Plan CUPED analysis during experiment design."""
    
    print("📋 CUPED PLANNING CHECKLIST:")
    print("=" * 40)
    
    # Check data availability
    available_covariates = [col for col in potential_covariates if col in historical_data.columns]
    print(f"✅ Available covariates: {len(available_covariates)}/{len(potential_covariates)}")
    
    # Estimate potential variance reduction
    for covar in available_covariates:
        corr = historical_data[covar].corr(historical_data[outcome_col])
        expected_reduction = corr**2  # Approximate variance reduction
        print(f"  {covar}: ~{expected_reduction:.1%} variance reduction")
    
    # Power calculation
    best_covariates = [col for col in available_covariates 
                      if abs(historical_data[col].corr(historical_data[outcome_col])) > 0.2]
    
    if best_covariates:
        print(f"🎯 Recommended for CUPED: {best_covariates}")
        print("💡 Expected benefits: 2-3× faster experiments")
    else:
        print("⚠️ Limited CUPED potential with current data")

# Use during experiment planning
plan_cuped_analysis(historical_data, 'conversion_rate', potential_covariates)
```

### 2. Robust Analysis Workflow

```python
def robust_cuped_analysis(data, outcome_col, treatment_col, covariate_cols):
    """Comprehensive CUPED analysis with error handling."""
    
    try:
        # Initialize
        cuped = CUPED(data, random_state=42)
        
        # Step 1: Validate data
        print("🔍 Data Validation...")
        assert outcome_col in data.columns, f"Outcome column '{outcome_col}' not found"
        assert treatment_col in data.columns, f"Treatment column '{treatment_col}' not found"
        missing_covariates = [col for col in covariate_cols if col not in data.columns]
        if missing_covariates:
            print(f"⚠️ Missing covariates: {missing_covariates}")
            covariate_cols = [col for col in covariate_cols if col in data.columns]
        
        # Step 2: Estimate effects
        print("📊 Estimating treatment effects...")
        results = cuped.estimate_treatment_effects(
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            covariate_cols=covariate_cols
        )
        
        # Step 3: Validate results
        print("✅ Validation...")
        variance_reduction = results['summary']['variance_reduction']
        if variance_reduction < 0:
            print("⚠️ Negative variance reduction - consider different covariates")
        
        # Step 4: Generate report
        print("📋 Generating report...")
        report = cuped.generate_summary_report(outcome_col)
        
        return results, report
        
    except Exception as e:
        print(f"❌ CUPED analysis failed: {e}")
        return None, None

# Use for robust analysis
results, report = robust_cuped_analysis(
    data=experiment_data,
    outcome_col='conversion_rate',
    treatment_col='treatment_group', 
    covariate_cols=['baseline_conversion', 'user_engagement']
)
```

### 3. Continuous Monitoring

```python
# Monitor CUPED performance across experiments
def track_cuped_performance(experiment_results):
    """Track CUPED effectiveness across multiple experiments."""
    
    performance_log = []
    
    for experiment_name, results in experiment_results.items():
        if results:
            performance_log.append({
                'experiment': experiment_name,
                'variance_reduction': results['summary']['variance_reduction'],
                'power_improvement': results['summary']['power_improvement'],
                'significance_gained': (
                    results['original']['p_value'] >= 0.05 and 
                    results['cuped']['p_value'] < 0.05
                )
            })
    
    # Analyze performance trends
    df = pd.DataFrame(performance_log)
    print("📈 CUPED Performance Tracking:")
    print(f"Average variance reduction: {df['variance_reduction'].mean():.1%}")
    print(f"Average power improvement: {df['power_improvement'].mean():.1f}×")
    print(f"Significance gained in {df['significance_gained'].sum()}/{len(df)} experiments")
    
    return df

# Track across experiments
performance_df = track_cuped_performance(all_experiment_results)
```

## 📚 Technical References

### Academic Foundation
- **Deng et al. (2013)**: "Improving the sensitivity of online controlled experiments by utilizing pre-experiment data" (Original CUPED paper)
- **Xu et al. (2015)**: "Leveraging machine learning to improve online controlled experiments"
- **Kohavi et al. (2020)**: "Trustworthy Online Controlled Experiments" (Chapter on variance reduction)

### Statistical Theory
CUPED is based on the **control variate method** from statistics:
- Reduces variance while maintaining unbiasedness
- Optimal coefficient minimizes mean squared error
- Effective for any pre-experiment predictor correlated with outcome

### When CUPED Applies
✅ **Perfect for:**
- Randomized controlled experiments (A/B tests)
- Rich pre-experiment user data available
- Need to increase experiment sensitivity
- Want faster time to statistical significance

❌ **Not suitable for:**
- Observational studies (use PSM/DML instead)
- No pre-experiment data available
- Treatment assignment not randomized

## 🤝 Integration Examples

### With A/B Testing Platforms

```python
# Integration with experiment tracking
def run_experiment_with_cuped(experiment_config):
    """Run A/B test with automatic CUPED analysis."""
    
    # Load experiment data
    experiment_data = load_experiment_data(experiment_config['experiment_id'])
    
    # Standard A/B test analysis
    standard_results = simple_ab_test(
        data=experiment_data,
        outcome=experiment_config['primary_metric'],
        treatment=experiment_config['treatment_column']
    )
    
    # CUPED analysis
    cuped = CUPED(experiment_data)
    cuped_results = cuped.estimate_treatment_effects(
        outcome_col=experiment_config['primary_metric'],
        treatment_col=experiment_config['treatment_column'],
        covariate_cols=experiment_config['cuped_covariates']
    )
    
    # Compare and report
    comparison = {
        'standard': standard_results,
        'cuped': cuped_results,
        'improvement': {
            'variance_reduction': cuped_results['summary']['variance_reduction'],
            'power_gain': cuped_results['summary']['power_improvement']
        }
    }
    
    return comparison

# Example usage
experiment_config = {
    'experiment_id': 'homepage_redesign_v1',
    'primary_metric': 'conversion_rate',
    'treatment_column': 'treatment_group',
    'cuped_covariates': ['baseline_conversion', 'user_engagement', 'sessions_last_week']
}

results = run_experiment_with_cuped(experiment_config)
```

### Business Reporting

```python
def generate_business_cuped_report(cuped_results, experiment_context):
    """Generate executive summary of CUPED analysis."""
    
    ate = cuped_results['cuped']['ate']
    ci_lower = cuped_results['cuped']['ci_lower'] 
    ci_upper = cuped_results['cuped']['ci_upper']
    p_value = cuped_results['cuped']['p_value']
    variance_reduction = cuped_results['summary']['variance_reduction']
    
    report = f"""
📊 EXPERIMENT RESULTS SUMMARY
{'='*50}

🎯 EXPERIMENT: {experiment_context['name']}
📅 PERIOD: {experiment_context['start_date']} to {experiment_context['end_date']}
👥 SAMPLE SIZE: {experiment_context['sample_size']:,} users

📈 TREATMENT EFFECT (CUPED-Adjusted):
• Impact: {ate:.1%} increase in {experiment_context['metric_name']}
• Confidence Interval: {ci_lower:.1%} to {ci_upper:.1%}
• Statistical Significance: {'✅ YES' if p_value < 0.05 else '❌ NO'} (p = {p_value:.4f})

🚀 CUPED METHODOLOGY BENEFITS:
• Variance Reduction: {variance_reduction:.1%}
• Analysis Precision: {(1-variance_reduction):.1%} more precise than standard A/B test
• Business Value: Earlier, more confident decision-making

💼 BUSINESS RECOMMENDATION:
"""
    
    if p_value < 0.05 and ate > 0:
        report += "✅ IMPLEMENT: Strong evidence of positive impact\n✅ SCALE: Roll out to full user base"
    elif p_value < 0.05 and ate < 0:
        report += "❌ DO NOT IMPLEMENT: Significant negative impact detected"
    else:
        report += "🔍 INCONCLUSIVE: Consider longer test period or larger sample"
    
    return report

# Generate executive report
business_report = generate_business_cuped_report(
    cuped_results=results,
    experiment_context={
        'name': 'Smart Filing Assistant Test',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'sample_size': 10000,
        'metric_name': 'Filing Completion Rate'
    }
)

print(business_report)
```

## 🎓 Learning Path

### Beginner (Start Here)
1. **Understand the concept**: CUPED = variance reduction for experiments
2. **Run the notebook**: `notebooks/04_cuped_tax.ipynb`
3. **Try basic analysis**: Single covariate, simple outcome
4. **Interpret results**: Focus on variance reduction metric

### Intermediate 
1. **Multiple covariates**: Experiment with different combinations
2. **Method comparison**: Try OLS, Ridge, Lasso adjustment methods
3. **Balance checking**: Understand covariate balance importance
4. **Business translation**: Connect statistical results to business impact

### Advanced
1. **Integration planning**: Build CUPED into experiment workflow
2. **Custom covariates**: Engineer features for better variance reduction
3. **Performance monitoring**: Track CUPED effectiveness over time
4. **Method extensions**: Contribute improvements to implementation

---

**🚀 Ready to get started?** Check out the interactive notebook at `notebooks/04_cuped_tax.ipynb` for a hands-on introduction to CUPED with real examples! 