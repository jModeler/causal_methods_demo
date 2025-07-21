# Difference-in-Differences (DiD) Guide

## 🎯 Overview

Difference-in-Differences (DiD) is a powerful causal inference technique designed for **panel data** where you observe the same units (users, companies, regions) over multiple time periods. DiD exploits the timing of treatment implementation to estimate causal effects by comparing changes in outcomes over time between treated and control groups.

## 🔬 Methodology

### Core Concept

DiD estimates the causal effect by calculating the "difference in differences":

```
DiD = [Treated_post - Treated_pre] - [Control_post - Control_pre]
```

This approach **controls for time-invariant confounders** and **common time trends**, making it particularly valuable when randomization isn't possible.

### Mathematical Foundation

The standard DiD regression model:

```
Y_it = α + β₁·Treated_i + β₂·Post_t + β₃·(Treated_i × Post_t) + X_it·γ + ε_it
```

Where:
- **Y_it** = Outcome for unit i at time t
- **Treated_i** = 1 if unit i receives treatment (time-invariant)
- **Post_t** = 1 if time period t is after treatment (time-variant)
- **β₃** = **DiD treatment effect** (coefficient of interest)
- **X_it** = Control variables
- **ε_it** = Error term

### Key Assumptions

1. **Parallel Trends**: Treatment and control groups would follow similar trends absent treatment
2. **No Anticipation**: Units don't change behavior before treatment starts
3. **Stable Unit Treatment Value (SUTVA)**: No spillover effects between units
4. **Treatment Timing**: Clear before/after periods

## 🚀 Key Features

### ✨ Robust Causal Identification
- **Controls for unobserved heterogeneity**: Time-invariant confounders eliminated
- **Handles time trends**: Common shocks and trends accounted for
- **Natural experiment design**: Exploits variation in treatment timing
- **Policy evaluation**: Ideal for analyzing interventions and feature rollouts

### 🔧 Advanced Implementation
- **Panel data preparation**: Automatic reshaping from wide to long format
- **Robust standard errors**: Clustered by unit to handle correlation
- **Parallel trends testing**: Validate key identification assumption
- **Heterogeneous effects**: Estimate effects by subgroups
- **Missing data handling**: Graceful degradation with incomplete panels

### 📊 Comprehensive Diagnostics
- **Parallel trends visualization**: Graphical assessment of assumptions
- **Balance checking**: Pre-treatment covariate balance
- **Robustness tests**: Placebo tests and sensitivity analysis
- **Statistical inference**: Confidence intervals, p-values, significance testing

## 📊 Usage Examples

### Basic DiD Analysis

```python
from src.causal_methods.did import DifferenceInDifferences

# Initialize DiD analyzer
did = DifferenceInDifferences(data)

# Prepare panel data (reshape from wide to long)
panel_df = did.prepare_panel_data(
    user_id_col='user_id',
    treatment_col='used_smart_assistant',
    outcome_2023_col='filed_2023',
    outcome_2024_col='filed_2024'
)

# Estimate basic DiD effect
results = did.estimate_did(
    outcome_col='outcome',
    cluster_se=True  # Cluster standard errors by user
)

# View results
print(f"DiD Treatment Effect: {results['did_estimate']:.4f}")
print(f"Standard Error: {results['standard_error']:.4f}")
print(f"P-value: {results['p_value']:.4f}")
print(f"95% CI: [{results['conf_int_lower']:.4f}, {results['conf_int_upper']:.4f}]")
```

### Advanced DiD with Controls

```python
# Estimate DiD with control variables
control_variables = ['age', 'income_bracket', 'tech_savviness', 'region']

controlled_results = did.estimate_did(
    outcome_col='outcome',
    control_vars=control_variables,
    cluster_se=True
)

print(f"Controlled DiD Effect: {controlled_results['did_estimate']:.4f}")
print(f"Formula used: {controlled_results['formula']}")
print(f"Sample size: {controlled_results['n_observations']} observations")
print(f"Number of users: {controlled_results['n_users']} users")

# Compare with basic model
improvement = abs(controlled_results['standard_error']) < abs(results['standard_error'])
print(f"Controls improved precision: {'✅ Yes' if improvement else '❌ No'}")
```

### Heterogeneous Treatment Effects

```python
# Analyze effect heterogeneity by user characteristics
subgroup_results = did.estimate_heterogeneous_effects(
    subgroup_var='age_group',  # or 'income_bracket', 'region', etc.
    outcome_col='outcome'
)

print(f"Effects by {subgroup_results['subgroup_variable']}:")
for subgroup, stats in subgroup_results['results'].items():
    significance = "✅ Significant" if stats['p_value'] < 0.05 else "❌ Not significant"
    print(f"  {subgroup}: {stats['did_estimate']:.4f} ({significance})")
    
# Visualize heterogeneous effects
fig = did.plot_subgroup_effects(subgroup_results, figsize=(12, 6))
plt.show()
```

### Parallel Trends Testing

```python
# Test the parallel trends assumption
trends_test = did.parallel_trends_test()

print("📊 PARALLEL TRENDS ASSESSMENT:")
print(f"Test type: {trends_test['test_type']}")
print(f"Description: {trends_test['description']}")

for outcome, stats in trends_test['results'].items():
    concern_level = "⚠️ High concern" if stats['significant'] else "✅ Low concern"
    print(f"  {outcome}: r={stats['correlation']:.3f}, p={stats['p_value']:.3f} ({concern_level})")

print(f"\n💡 Interpretation: {trends_test['interpretation']}")

# Visualize parallel trends
fig = did.plot_parallel_trends(
    outcome_2023_col='filed_2023',
    outcome_2024_col='filed_2024',
    figsize=(10, 6)
)
plt.show()
```

### Complete DiD Workflow

```python
def comprehensive_did_analysis(data, treatment_col, outcome_cols, control_vars=None):
    """Complete DiD analysis workflow with all diagnostics."""
    
    print("🔍 DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 50)
    
    # Initialize
    did = DifferenceInDifferences(data)
    
    # Prepare panel data
    print("📊 Preparing panel data...")
    panel_df = did.prepare_panel_data(
        treatment_col=treatment_col,
        outcome_2023_col=outcome_cols[0], 
        outcome_2024_col=outcome_cols[1]
    )
    
    print(f"Panel observations: {len(panel_df)}")
    print(f"Unique users: {panel_df['user_id'].nunique()}")
    print(f"Time periods: {sorted(panel_df['year'].unique())}")
    
    # Basic DiD estimation
    print("\n🎯 Estimating treatment effects...")
    basic_results = did.estimate_did()
    
    # With controls (if provided)
    if control_vars:
        controlled_results = did.estimate_did(control_vars=control_vars)
        print("✅ Analysis completed with control variables")
    else:
        controlled_results = basic_results
        print("✅ Basic analysis completed")
    
    # Parallel trends test
    print("\n🔍 Testing parallel trends assumption...")
    trends_test = did.parallel_trends_test()
    
    # Summary
    print(f"\n📈 FINAL RESULTS:")
    print(f"DiD Treatment Effect: {controlled_results['did_estimate']:.4f}")
    print(f"Standard Error: {controlled_results['standard_error']:.4f}")
    significance = "✅ Significant" if controlled_results['p_value'] < 0.05 else "❌ Not significant"
    print(f"Statistical Significance: {significance} (p = {controlled_results['p_value']:.4f})")
    
    return {
        'panel_data': panel_df,
        'basic_results': basic_results,
        'controlled_results': controlled_results,
        'trends_test': trends_test,
        'did_object': did
    }

# Run comprehensive analysis
analysis_results = comprehensive_did_analysis(
    data=tax_software_data,
    treatment_col='used_smart_assistant',
    outcome_cols=['filed_2023', 'filed_2024'],
    control_vars=['age', 'tech_savviness', 'income_bracket']
)
```

## 🎛️ Configuration Options

### Standard Error Options

| Method | Use Case | Description |
|--------|----------|-------------|
| `cluster_se=True` | **Recommended** | Cluster by unit (user) to handle within-unit correlation |
| `cluster_se=False` | Simple cases | Standard OLS standard errors |

```python
# Compare standard error methods
results_clustered = did.estimate_did(cluster_se=True)
results_standard = did.estimate_did(cluster_se=False)

print("Standard Error Comparison:")
print(f"Clustered SE: {results_clustered['standard_error']:.4f}")
print(f"Standard SE: {results_standard['standard_error']:.4f}")

# Clustered SEs are typically larger and more conservative
robustness = results_clustered['standard_error'] > results_standard['standard_error']
print(f"Clustered SEs more conservative: {'✅ Yes' if robustness else '❌ No'}")
```

### Panel Data Preparation

```python
# Customize panel data preparation
panel_df = did.prepare_panel_data(
    user_id_col='customer_id',         # Custom user identifier
    treatment_col='feature_adoption',   # Custom treatment variable
    outcome_2023_col='baseline_metric', # Custom pre-period outcome
    outcome_2024_col='current_metric'   # Custom post-period outcome
)

# Check panel structure
print("Panel Data Structure:")
print(f"Total observations: {len(panel_df)}")
print(f"Users per period: {panel_df.groupby('year')['user_id'].nunique().to_dict()}")
print(f"Treatment rates: {panel_df.groupby('year')['treated'].mean().to_dict()}")
```

## 📈 Identifying Good DiD Opportunities

### What Makes DiD Effective?

1. **Clear treatment timing**: Definite before/after periods
2. **Stable control group**: Units never receive treatment
3. **Parallel pre-trends**: Similar trajectories before treatment
4. **Sufficient time periods**: Multiple observations per unit
5. **Relevant control variables**: Reduce remaining confounding

### DiD Suitability Assessment

```python
def assess_did_suitability(data, treatment_col, outcome_cols, user_id_col='user_id'):
    """Assess whether DiD is appropriate for your data."""
    
    print("🔍 DiD SUITABILITY ASSESSMENT")
    print("=" * 40)
    
    # Check basic requirements
    has_panel = user_id_col in data.columns
    has_treatment = treatment_col in data.columns
    has_outcomes = all(col in data.columns for col in outcome_cols)
    
    print(f"✅ Panel structure: {'Yes' if has_panel else '❌ No'}")
    print(f"✅ Treatment variable: {'Yes' if has_treatment else '❌ No'}")
    print(f"✅ Outcome variables: {'Yes' if has_outcomes else '❌ No'}")
    
    if not all([has_panel, has_treatment, has_outcomes]):
        print("❌ Basic requirements not met for DiD")
        return False
    
    # Check treatment variation
    treatment_rate = data[treatment_col].mean()
    print(f"Treatment rate: {treatment_rate:.1%}")
    
    if treatment_rate == 0 or treatment_rate == 1:
        print("❌ No treatment variation - DiD not applicable")
        return False
    
    # Check temporal variation
    pre_outcome_var = data[outcome_cols[0]].var()
    post_outcome_var = data[outcome_cols[1]].var()
    print(f"Pre-period outcome variance: {pre_outcome_var:.3f}")
    print(f"Post-period outcome variance: {post_outcome_var:.3f}")
    
    # Check for potential parallel trends (basic check)
    treated_change = (
        data[data[treatment_col] == 1][outcome_cols[1]].mean() - 
        data[data[treatment_col] == 1][outcome_cols[0]].mean()
    )
    control_change = (
        data[data[treatment_col] == 0][outcome_cols[1]].mean() - 
        data[data[treatment_col] == 0][outcome_cols[0]].mean()
    )
    
    print(f"Treated group change: {treated_change:.4f}")
    print(f"Control group change: {control_change:.4f}")
    print(f"Naive DiD estimate: {treated_change - control_change:.4f}")
    
    # Overall assessment
    suitable = (0.1 <= treatment_rate <= 0.9 and 
               pre_outcome_var > 0 and post_outcome_var > 0)
    
    print(f"\n🎯 DiD Suitability: {'✅ Good' if suitable else '⚠️ Questionable'}")
    
    if suitable:
        print("💡 Recommendation: Proceed with DiD analysis")
    else:
        print("💡 Recommendation: Consider PSM or DML instead")
    
    return suitable

# Assess your data
is_suitable = assess_did_suitability(
    data=your_data,
    treatment_col='used_smart_assistant',
    outcome_cols=['filed_2023', 'filed_2024']
)
```

### Common DiD Use Cases

#### 1. **Feature Rollouts**
```python
# Analyze impact of feature launch
feature_did = DifferenceInDifferences(feature_rollout_data)

# Users who adopted new feature vs. those who didn't
panel_df = feature_did.prepare_panel_data(
    treatment_col='adopted_new_feature',
    outcome_2023_col='engagement_before',
    outcome_2024_col='engagement_after'
)
```

#### 2. **Geographic Interventions**
```python
# Compare regions with/without intervention
geo_did = DifferenceInDifferences(regional_data)

# Regions that received intervention vs. control regions
panel_df = geo_did.prepare_panel_data(
    user_id_col='region_id',
    treatment_col='received_intervention',
    outcome_2023_col='baseline_outcome',
    outcome_2024_col='post_outcome'
)
```

#### 3. **Policy Changes**
```python
# Analyze policy impact on different user segments
policy_did = DifferenceInDifferences(policy_data)

# Users affected by policy vs. unaffected users
panel_df = policy_did.prepare_panel_data(
    treatment_col='affected_by_policy',
    outcome_2023_col='pre_policy_behavior',
    outcome_2024_col='post_policy_behavior'
)
```

## 🔍 Interpreting DiD Results

### Statistical Significance Assessment

```python
def interpret_did_results(results, effect_size_thresholds=None):
    """Interpret DiD results for business decision-making."""
    
    if effect_size_thresholds is None:
        effect_size_thresholds = {'small': 0.01, 'medium': 0.05, 'large': 0.10}
    
    did_estimate = results['did_estimate']
    p_value = results['p_value']
    ci_lower = results['conf_int_lower']
    ci_upper = results['conf_int_upper']
    
    print("📊 DiD RESULTS INTERPRETATION")
    print("=" * 40)
    
    # Statistical significance
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not statistically significant"
    
    print(f"Statistical Significance: {significance}")
    
    # Effect size interpretation
    abs_effect = abs(did_estimate)
    if abs_effect >= effect_size_thresholds['large']:
        effect_size = "large"
    elif abs_effect >= effect_size_thresholds['medium']:
        effect_size = "medium"
    elif abs_effect >= effect_size_thresholds['small']:
        effect_size = "small"
    else:
        effect_size = "negligible"
    
    direction = "positive" if did_estimate > 0 else "negative"
    print(f"Effect Size: {effect_size} {direction} effect ({did_estimate:.4f})")
    
    # Confidence interval assessment
    ci_width = ci_upper - ci_lower
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Confidence Interval Width: {ci_width:.4f}")
    
    # Practical significance
    both_positive = ci_lower > 0 and ci_upper > 0
    both_negative = ci_lower < 0 and ci_upper < 0
    
    if both_positive:
        practical = "consistently positive effect"
    elif both_negative:
        practical = "consistently negative effect"
    else:
        practical = "effect direction uncertain"
    
    print(f"Practical Significance: {practical}")
    
    # Business recommendation
    print(f"\n🎯 BUSINESS RECOMMENDATION:")
    if p_value < 0.05 and abs_effect >= effect_size_thresholds['small']:
        if did_estimate > 0:
            print("✅ POSITIVE IMPACT: Consider expanding intervention")
        else:
            print("❌ NEGATIVE IMPACT: Reconsider intervention")
    else:
        print("🔍 INCONCLUSIVE: Need more data or longer observation period")

# Apply to your results
interpret_did_results(
    results=controlled_results,
    effect_size_thresholds={'small': 0.02, 'medium': 0.05, 'large': 0.10}
)
```

### Business Impact Translation

```python
def translate_to_business_metrics(did_results, business_context):
    """Translate DiD statistical results to business impact."""
    
    did_effect = did_results['did_estimate']
    ci_lower = did_results['conf_int_lower']
    ci_upper = did_results['conf_int_upper']
    
    baseline_rate = business_context.get('baseline_rate', 0.5)
    total_users = business_context.get('total_users', 100000)
    
    print("💼 BUSINESS IMPACT TRANSLATION")
    print("=" * 40)
    
    # Calculate business metrics
    if business_context['outcome_type'] == 'binary':
        # For binary outcomes (conversion, filing, etc.)
        absolute_improvement = did_effect
        relative_improvement = did_effect / baseline_rate
        
        users_affected = total_users * did_effect
        
        print(f"Outcome: {business_context['outcome_name']}")
        print(f"Baseline Rate: {baseline_rate:.1%}")
        print(f"Treatment Effect: {absolute_improvement:.1%} (absolute)")
        print(f"Relative Improvement: {relative_improvement:.1%}")
        print(f"Users Affected: {users_affected:,.0f}")
        
        # Confidence intervals for business metrics
        users_affected_lower = total_users * ci_lower
        users_affected_upper = total_users * ci_upper
        print(f"Impact Range: {users_affected_lower:,.0f} to {users_affected_upper:,.0f} users")
        
    elif business_context['outcome_type'] == 'continuous':
        # For continuous outcomes (revenue, time, etc.)
        unit = business_context.get('unit', 'units')
        
        print(f"Average Impact: {did_effect:.2f} {unit} per user")
        print(f"Total Impact: {did_effect * total_users:,.0f} {unit}")
        print(f"Impact Range: {ci_lower * total_users:,.0f} to {ci_upper * total_users:,.0f} {unit}")
    
    # ROI calculation (if cost data available)
    if 'intervention_cost' in business_context:
        cost = business_context['intervention_cost']
        benefit = users_affected * business_context.get('value_per_user', 10)
        roi = (benefit - cost) / cost if cost > 0 else float('inf')
        
        print(f"\n💰 ROI ANALYSIS:")
        print(f"Intervention Cost: ${cost:,.0f}")
        print(f"Estimated Benefit: ${benefit:,.0f}")
        print(f"ROI: {roi:.1%}")

# Example usage
business_context = {
    'outcome_type': 'binary',
    'outcome_name': 'Tax Filing Completion',
    'baseline_rate': 0.85,
    'total_users': 50000,
    'value_per_user': 25,
    'intervention_cost': 100000
}

translate_to_business_metrics(controlled_results, business_context)
```

## ⚠️ Common Pitfalls & Solutions

### 1. Parallel Trends Violation

**Problem**: Treatment and control groups have different pre-trends
```python
# Diagnose parallel trends violation
trends_test = did.parallel_trends_test()

violations = [outcome for outcome, stats in trends_test['results'].items() 
             if stats['significant']]

if violations:
    print(f"⚠️ Parallel trends concerns: {violations}")
    print("💡 Solutions:")
    print("  - Add more control variables")
    print("  - Use matching before DiD")
    print("  - Consider alternative identification strategies")
```

**Solutions**:
- Add more control variables
- Use propensity score matching before DiD
- Implement triple-differences approach
- Use synthetic control methods

### 2. Treatment Anticipation

**Problem**: Units change behavior before official treatment
```python
# Check for anticipation effects
def check_anticipation_effects(data, treatment_col, pre_outcomes):
    """Check if treatment assignment predicts pre-treatment changes."""
    
    for outcome in pre_outcomes:
        if outcome in data.columns:
            # Correlation between treatment and pre-period outcomes
            corr = data[treatment_col].corr(data[outcome])
            
            if abs(corr) > 0.1:
                print(f"⚠️ Possible anticipation in {outcome}: r = {corr:.3f}")
            else:
                print(f"✅ No anticipation detected in {outcome}: r = {corr:.3f}")

# Check your data
check_anticipation_effects(
    data=tax_data,
    treatment_col='used_smart_assistant',
    pre_outcomes=['filed_2023', 'sessions_2023']
)
```

### 3. Insufficient Time Periods

**Problem**: Only two time periods limit identification power
```python
# Assess temporal coverage
def assess_temporal_coverage(panel_data):
    """Check if temporal coverage is sufficient for robust DiD."""
    
    periods = panel_data['year'].nunique()
    users_per_period = panel_data.groupby('year')['user_id'].nunique()
    
    print(f"Time periods: {periods}")
    print(f"Users per period: {users_per_period.to_dict()}")
    
    if periods < 3:
        print("⚠️ Limited time periods - consider longer observation window")
    else:
        print("✅ Adequate temporal coverage")
    
    return periods >= 3

# Check your panel data
adequate_coverage = assess_temporal_coverage(panel_df)
```

### 4. Treatment Group Contamination

**Problem**: Control group receives partial treatment
```python
# Check treatment group purity
def check_treatment_purity(data, treatment_col):
    """Verify clean treatment and control groups."""
    
    treatment_dist = data[treatment_col].value_counts(normalize=True)
    print("Treatment Distribution:")
    print(treatment_dist)
    
    # Check for middle values (should be 0 or 1)
    intermediate_values = data[treatment_col].unique()
    intermediate_values = intermediate_values[(intermediate_values != 0) & (intermediate_values != 1)]
    
    if len(intermediate_values) > 0:
        print(f"⚠️ Intermediate treatment values detected: {intermediate_values}")
        print("💡 Consider recoding or excluding these observations")
    else:
        print("✅ Clean binary treatment assignment")

check_treatment_purity(data, 'used_smart_assistant')
```

## 🎯 Best Practices

### 1. Pre-Analysis Planning

```python
def plan_did_analysis(data, potential_outcomes, potential_treatments):
    """Plan DiD analysis before implementation."""
    
    print("📋 DiD ANALYSIS PLANNING")
    print("=" * 40)
    
    # Check data structure
    print("🔍 Data Structure Assessment:")
    print(f"Sample size: {len(data):,} observations")
    print(f"Potential outcomes: {potential_outcomes}")
    print(f"Potential treatments: {potential_treatments}")
    
    # Assess each treatment-outcome combination
    for treatment in potential_treatments:
        for outcome_pair in potential_outcomes:
            if all(col in data.columns for col in [treatment] + outcome_pair):
                
                treatment_rate = data[treatment].mean()
                pre_var = data[outcome_pair[0]].var()
                post_var = data[outcome_pair[1]].var()
                
                print(f"\n📊 {treatment} → {outcome_pair}:")
                print(f"  Treatment rate: {treatment_rate:.1%}")
                print(f"  Pre-period variance: {pre_var:.3f}")
                print(f"  Post-period variance: {post_var:.3f}")
                
                # Quick feasibility check
                feasible = (0.05 <= treatment_rate <= 0.95 and 
                           pre_var > 0 and post_var > 0)
                print(f"  Feasibility: {'✅ Good' if feasible else '⚠️ Concerning'}")

# Plan your analysis
plan_did_analysis(
    data=your_data,
    potential_outcomes=[['filed_2023', 'filed_2024'], ['revenue_2023', 'revenue_2024']],
    potential_treatments=['used_smart_assistant', 'received_promotion']
)
```

### 2. Robust Analysis Workflow

```python
def robust_did_workflow(data, treatment_col, outcome_cols, control_vars=None):
    """Comprehensive DiD analysis with robustness checks."""
    
    results = {}
    
    try:
        # Step 1: Initialize and prepare data
        did = DifferenceInDifferences(data)
        panel_df = did.prepare_panel_data(
            treatment_col=treatment_col,
            outcome_2023_col=outcome_cols[0],
            outcome_2024_col=outcome_cols[1]
        )
        results['panel_prepared'] = True
        
        # Step 2: Basic estimation
        basic_results = did.estimate_did()
        results['basic_estimation'] = basic_results
        
        # Step 3: Controlled estimation
        if control_vars:
            controlled_results = did.estimate_did(control_vars=control_vars)
            results['controlled_estimation'] = controlled_results
        
        # Step 4: Parallel trends test
        trends_test = did.parallel_trends_test()
        results['trends_test'] = trends_test
        
        # Step 5: Heterogeneity analysis
        if 'age' in data.columns:
            subgroup_results = did.estimate_heterogeneous_effects('age_group')
            results['heterogeneity'] = subgroup_results
        
        # Step 6: Robustness summary
        robust_estimate = controlled_results if control_vars else basic_results
        results['final_results'] = robust_estimate
        results['success'] = True
        
        print("✅ Robust DiD analysis completed successfully")
        
    except Exception as e:
        print(f"❌ DiD analysis failed: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results

# Run robust analysis
comprehensive_results = robust_did_workflow(
    data=tax_software_data,
    treatment_col='used_smart_assistant',
    outcome_cols=['filed_2023', 'filed_2024'],
    control_vars=['age', 'tech_savviness', 'region']
)
```

### 3. Sensitivity Analysis

```python
def did_sensitivity_analysis(data, treatment_col, outcome_cols):
    """Test sensitivity of DiD results to various specifications."""
    
    print("🔬 DiD SENSITIVITY ANALYSIS")
    print("=" * 40)
    
    did = DifferenceInDifferences(data)
    panel_df = did.prepare_panel_data(
        treatment_col=treatment_col,
        outcome_2023_col=outcome_cols[0],
        outcome_2024_col=outcome_cols[1]
    )
    
    # Test different specifications
    specifications = {
        'basic': [],
        'demographic': ['age', 'income_bracket'],
        'behavioral': ['tech_savviness', 'user_type'],
        'full': ['age', 'income_bracket', 'tech_savviness', 'user_type', 'region']
    }
    
    sensitivity_results = {}
    
    for spec_name, controls in specifications.items():
        # Filter controls that exist in data
        available_controls = [c for c in controls if c in data.columns]
        
        try:
            if available_controls:
                results = did.estimate_did(control_vars=available_controls)
            else:
                results = did.estimate_did()
            
            sensitivity_results[spec_name] = {
                'estimate': results['did_estimate'],
                'se': results['standard_error'],
                'p_value': results['p_value'],
                'controls': available_controls
            }
            
        except Exception as e:
            print(f"⚠️ Specification '{spec_name}' failed: {e}")
    
    # Compare results
    print("\nSensitivity Comparison:")
    for spec, stats in sensitivity_results.items():
        significance = "✅ Sig" if stats['p_value'] < 0.05 else "❌ Not sig"
        print(f"  {spec}: {stats['estimate']:.4f} (SE: {stats['se']:.4f}) {significance}")
    
    # Check robustness
    estimates = [stats['estimate'] for stats in sensitivity_results.values()]
    estimate_range = max(estimates) - min(estimates)
    
    print(f"\nRobustness Assessment:")
    print(f"Estimate range: {estimate_range:.4f}")
    print(f"Robust to specification: {'✅ Yes' if estimate_range < 0.01 else '⚠️ Questionable'}")
    
    return sensitivity_results

# Run sensitivity analysis
sensitivity_results = did_sensitivity_analysis(
    data=tax_data,
    treatment_col='used_smart_assistant',
    outcome_cols=['filed_2023', 'filed_2024']
)
```

## 📚 Technical References

### Academic Foundation
- **Angrist & Pischke (2009)**: "Mostly Harmless Econometrics" - Chapter 5 on DiD
- **Card & Krueger (1994)**: Classic minimum wage study using DiD
- **Bertrand et al. (2004)**: "How Much Should We Trust Differences-in-Differences Estimates?"
- **Goodman-Bacon (2021)**: "Difference-in-differences with variation in treatment timing"

### Statistical Theory
DiD is based on the **identifying assumption** that:
```
E[Y₁ᵢₜ - Y₀ᵢₜ | Dᵢ = 1] = E[Y₁ᵢₜ - Y₀ᵢₜ | Dᵢ = 0]
```

Where parallel trends ensures that treatment effects are identified from variation in treatment timing rather than permanent differences between groups.

### When DiD Applies
✅ **Perfect for:**
- Feature rollouts with clear timing
- Geographic interventions 
- Policy changes affecting specific groups
- Natural experiments with treatment timing variation

❌ **Not suitable for:**
- Cross-sectional data (no time dimension)
- Universal treatment (no control group)
- Continuous treatment assignment
- Short observation windows

## 🤝 Integration Examples

### With A/B Testing

```python
def did_for_ab_testing(experiment_data, feature_launch_date):
    """Use DiD to analyze long-term effects of A/B test features."""
    
    # Split data into pre/post launch periods
    pre_data = experiment_data[experiment_data['date'] < feature_launch_date]
    post_data = experiment_data[experiment_data['date'] >= feature_launch_date]
    
    # Aggregate to user level
    user_pre = pre_data.groupby('user_id').agg({
        'conversion': 'mean',
        'revenue': 'sum',
        'treatment_group': 'first'
    }).rename(columns={'conversion': 'conversion_pre', 'revenue': 'revenue_pre'})
    
    user_post = post_data.groupby('user_id').agg({
        'conversion': 'mean', 
        'revenue': 'sum'
    }).rename(columns={'conversion': 'conversion_post', 'revenue': 'revenue_post'})
    
    # Combine for DiD analysis
    did_data = user_pre.merge(user_post, left_index=True, right_index=True)
    
    # Run DiD
    did = DifferenceInDifferences(did_data.reset_index())
    panel_df = did.prepare_panel_data(
        user_id_col='user_id',
        treatment_col='treatment_group',
        outcome_2023_col='conversion_pre',
        outcome_2024_col='conversion_post'
    )
    
    return did.estimate_did()

# Apply to A/B test data
ab_did_results = did_for_ab_testing(ab_test_data, '2024-06-01')
```

### Business Reporting

```python
def generate_did_business_report(did_results, business_context):
    """Generate executive summary of DiD analysis."""
    
    effect = did_results['did_estimate']
    se = did_results['standard_error']
    p_value = did_results['p_value']
    ci_lower = did_results['conf_int_lower']
    ci_upper = did_results['conf_int_upper']
    
    report = f"""
📊 DIFFERENCE-IN-DIFFERENCES ANALYSIS REPORT
{'='*60}

🎯 INTERVENTION: {business_context['intervention_name']}
📅 ANALYSIS PERIOD: {business_context['period']}
👥 SAMPLE: {business_context['sample_description']}

📈 CAUSAL IMPACT ESTIMATE:
• Treatment Effect: {effect:.1%} change in {business_context['outcome_name']}
• Statistical Precision: ±{se:.1%} (standard error)
• 95% Confidence Interval: {ci_lower:.1%} to {ci_upper:.1%}
• Statistical Significance: {'✅ YES' if p_value < 0.05 else '❌ NO'} (p = {p_value:.4f})

🔬 METHODOLOGY STRENGTHS:
• Controls for time-invariant user characteristics
• Accounts for common time trends and seasonal effects
• Exploits natural variation in intervention timing
• Robust to selection bias and unobserved confounding

💼 BUSINESS IMPLICATIONS:
"""
    
    if p_value < 0.05:
        if effect > 0:
            report += f"""✅ POSITIVE IMPACT CONFIRMED
• Intervention increases {business_context['outcome_name']} by {effect:.1%}
• Effect is statistically significant and economically meaningful
• Recommend continuation and potential expansion"""
        else:
            report += f"""❌ NEGATIVE IMPACT DETECTED  
• Intervention decreases {business_context['outcome_name']} by {abs(effect):.1%}
• Effect is statistically significant - immediate review needed
• Recommend intervention modification or discontinuation"""
    else:
        report += f"""🔍 INCONCLUSIVE RESULTS
• No statistically significant effect detected
• True effect likely between {ci_lower:.1%} and {ci_upper:.1%}
• Consider longer observation period or larger sample size"""
    
    report += f"""

🎯 NEXT STEPS:
• Monitor effect sustainability over longer time horizon
• Analyze heterogeneous effects across user segments  
• Conduct cost-benefit analysis of intervention continuation
• Consider replication with different populations or contexts
"""
    
    return report

# Generate business report
business_context = {
    'intervention_name': 'Smart Filing Assistant Feature',
    'period': 'January 2023 - January 2024',
    'sample_description': '10,000 tax software users',
    'outcome_name': 'Tax Filing Completion Rate'
}

business_report = generate_did_business_report(controlled_results, business_context)
print(business_report)
```

## 🎓 Learning Path

### Beginner (Start Here)
1. **Understand the intuition**: DiD = treatment timing variation
2. **Run the notebook**: `notebooks/01_did_tax.ipynb`
3. **Practice interpretation**: Focus on parallel trends concept
4. **Try basic analysis**: Single outcome, simple specification

### Intermediate
1. **Add control variables**: Improve precision with covariates
2. **Test robustness**: Multiple specifications and sensitivity analysis
3. **Heterogeneity analysis**: Explore treatment effect variation
4. **Visualization mastery**: Create compelling parallel trends plots

### Advanced
1. **Complex designs**: Multiple time periods, staggered adoption
2. **Integration strategies**: Combine with other causal methods
3. **Business applications**: Translate results to actionable insights
4. **Methodological extensions**: Event studies, synthetic controls

---

**🚀 Ready to get started?** Check out the interactive notebook at `notebooks/01_did_tax.ipynb` for a hands-on introduction to DiD with real examples! 