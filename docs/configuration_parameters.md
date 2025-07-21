# Configuration Parameters Reference

This document provides a comprehensive reference for all parameters in the synthetic data simulation configuration system.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Simulation Settings](#simulation-settings)
3. [Demographics](#demographics)
4. [Tech-Savviness](#tech-savviness)
5. [Baseline Behavior (2023)](#baseline-behavior-2023)
6. [Early Login Behavior](#early-login-behavior)
7. [Treatment Assignment](#treatment-assignment)
8. [Outcomes (2024)](#outcomes-2024)
9. [Output Settings](#output-settings)
10. [Parameter Interactions](#parameter-interactions)
11. [Creating Custom Scenarios](#creating-custom-scenarios)

---

## Overview

The configuration system uses YAML files to control all aspects of synthetic data generation for the tax software causal inference analysis. This eliminates magic numbers and makes experimentation easy.

### Key Principles
- **Inheritance**: Scenario configs inherit from base config
- **Override-only**: Only specify parameters you want to change
- **Self-documenting**: Each parameter includes comments explaining its purpose
- **Realistic ranges**: All parameters based on realistic business assumptions

---

## Simulation Settings

Controls basic simulation parameters and reproducibility.

```yaml
simulation:
  random_seed: 42              # Random seed for reproducibility
  default_n_users: 10000       # Default number of users to generate
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | int | 42 | Sets numpy and random seeds for reproducible results |
| `default_n_users` | int | 10000 | Default sample size when not specified in code |

### Impact
- **Random seed**: Ensures identical datasets across runs for reproducibility
- **Sample size**: Affects statistical power and runtime performance

---

## Demographics

Controls the distribution of user demographic characteristics.

```yaml
demographics:
  income_brackets:
    values: ["<30k", "30k-50k", "50k-75k", "75k-100k", "100k-150k", ">150k"]
    weights: [0.15, 0.20, 0.25, 0.20, 0.15, 0.05]
  
  device_types:
    values: ["mobile", "desktop", "tablet"]
    weights: [0.45, 0.50, 0.05]
  
  user_types:
    values: ["new", "returning"]
    weights: [0.35, 0.65]
  
  regions:
    values: ["West", "East", "Midwest", "South"]
    weights: [0.25, 0.30, 0.20, 0.25]
  
  age:
    mean: 45                   # Average user age
    std: 15                    # Standard deviation of age
    min_age: 18               # Minimum age (hard bound)
    max_age: 80               # Maximum age (hard bound)
```

### Income Brackets

| Parameter | Type | Description | Business Rationale |
|-----------|------|-------------|-------------------|
| `values` | list[str] | Income bracket labels | Standard income categorization |
| `weights` | list[float] | Probability weights (must sum to 1.0) | Based on US income distribution |

**Default Distribution**:
- `<30k`: 15% (Lower income, may have simpler tax situations)
- `30k-50k`: 20% (Lower-middle income)
- `50k-75k`: 25% (Middle income, largest segment)
- `75k-100k`: 20% (Upper-middle income)
- `100k-150k`: 15% (High income, more complex taxes)
- `>150k`: 5% (Very high income, complex situations)

### Device Types

| Parameter | Type | Description | Business Impact |
|-----------|------|-------------|-----------------|
| `values` | list[str] | Device type labels | Affects user experience and feature adoption |
| `weights` | list[float] | Usage probability | Mobile-first reflects modern usage patterns |

**Default Distribution**:
- `mobile`: 45% (Growing segment, different UX constraints)
- `desktop`: 50% (Traditional platform, full functionality)
- `tablet`: 5% (Niche usage, hybrid experience)

### User Types

| Parameter | Type | Description | Business Impact |
|-----------|------|-------------|-----------------|
| `values` | list[str] | User classification | Affects baseline behavior and feature adoption |
| `weights` | list[float] | Distribution probability | Returning users have established patterns |

**Default Distribution**:
- `new`: 35% (First-time users, need guidance)
- `returning`: 65% (Experienced users, established workflows)

### Geographic Regions

| Parameter | Type | Description | Business Impact |
|-----------|------|-------------|-----------------|
| `values` | list[str] | US geographic regions | Affects tech adoption and behavior patterns |
| `weights` | list[float] | Population distribution | Loosely based on US regional populations |

### Age Distribution

| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|---------|
| `mean` | float | 45 | Average user age | Core demographic for tax software |
| `std` | float | 15 | Age standard deviation | Creates realistic age spread |
| `min_age` | int | 18 | Minimum age cutoff | Legal/practical constraint |
| `max_age` | int | 80 | Maximum age cutoff | Practical constraint for technology use |

---

## Tech-Savviness

Controls the hidden confounder that affects both treatment adoption and outcomes.

```yaml
tech_savviness:
  base_score: 50               # Baseline tech-savviness (0-100 scale)
  
  age_adjustments:
    young_threshold: 35        # Age below which users get tech boost
    young_boost: 20           # Tech score boost for young users
    old_threshold: 55         # Age above which users get tech penalty
    old_penalty: -15          # Tech score penalty for older users
  
  income_adjustments:
    high_income_brackets: [">150k", "100k-150k"]
    high_income_boost: 15     # Tech boost for high income
    low_income_brackets: ["<30k"]
    low_income_penalty: -10   # Tech penalty for low income
  
  region_adjustments:
    west_boost: 10            # Tech boost for West Coast
    midwest_south_penalty: -5 # Tech penalty for Midwest/South
  
  std: 15                     # Standard deviation of tech scores
  min_score: 0               # Minimum tech score
  max_score: 100             # Maximum tech score
```

### Base Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_score` | int | 50 | Starting tech-savviness before adjustments |
| `std` | int | 15 | Variability in tech scores |
| `min_score` | int | 0 | Floor for tech-savviness |
| `max_score` | int | 100 | Ceiling for tech-savviness |

### Age-Based Adjustments

| Parameter | Type | Default | Rationale |
|-----------|------|---------|-----------|
| `young_threshold` | int | 35 | Digital natives cutoff |
| `young_boost` | int | 20 | Younger users more tech-comfortable |
| `old_threshold` | int | 55 | Traditional technology adoption cutoff |
| `old_penalty` | int | -15 | Older users may be less tech-comfortable |

### Income-Based Adjustments

| Parameter | Type | Default | Rationale |
|-----------|------|---------|-----------|
| `high_income_brackets` | list[str] | [">150k", "100k-150k"] | Higher income correlates with tech access |
| `high_income_boost` | int | 15 | Better devices, more tech exposure |
| `low_income_brackets` | list[str] | ["<30k"] | Limited tech access/exposure |
| `low_income_penalty` | int | -10 | Digital divide effects |

### Regional Adjustments

| Parameter | Type | Default | Rationale |
|-----------|------|---------|-----------|
| `west_boost` | int | 10 | Tech hub concentration (Silicon Valley, Seattle) |
| `midwest_south_penalty` | int | -5 | More traditional/rural areas |

### Tech-Savviness Impact

The tech-savviness score is the **primary driver of selection bias** in our simulation:
- **Treatment adoption**: Higher tech users more likely to try new features
- **Baseline efficiency**: Affects time to complete, support needs
- **Outcome heterogeneity**: Treatment effects vary by tech level

---

## Baseline Behavior (2023)

Controls pre-treatment behavior patterns that establish user baselines.

### Filing Behavior

```yaml
baseline_2023:
  filing:
    base_rate: 0.75           # Base probability of filing taxes
    
    income_effects:
      high_income_boost: 0.10  # Filing boost for 75k+ income
      low_income_penalty: -0.15 # Filing penalty for <30k income
    
    returning_user_boost: 0.08 # Returning users more likely to file
    
    prime_age_min: 25         # Start of prime filing age
    prime_age_max: 55         # End of prime filing age  
    prime_age_boost: 0.05     # Filing boost for prime age
    non_prime_penalty: -0.05  # Filing penalty outside prime age
```

#### Filing Parameters

| Parameter | Type | Default | Description | Business Logic |
|-----------|------|---------|-------------|----------------|
| `base_rate` | float | 0.75 | Base filing probability | ~75% filing rate is realistic |
| `high_income_boost` | float | 0.10 | Boost for 75k+ earners | Higher earners must file |
| `low_income_penalty` | float | -0.15 | Penalty for <30k earners | May not need to file |
| `returning_user_boost` | float | 0.08 | Boost for returning users | Established filing habit |
| `prime_age_boost` | float | 0.05 | Boost for ages 25-55 | Peak earning/filing years |
| `non_prime_penalty` | float | -0.05 | Penalty outside prime age | Students, retirees less likely |

### Time to Complete

```yaml
time_to_complete:
  base_time: 120              # Base completion time (minutes)
  returning_user_reduction: -30 # Time reduction for returning users
  high_tech_reduction: -20    # Time reduction for tech_savviness > 70
  low_tech_penalty: 40        # Time penalty for tech_savviness < 30
  std: 30                     # Standard deviation of completion times
  min_time: 30               # Minimum completion time
```

#### Time Parameters

| Parameter | Type | Default | Description | Business Logic |
|-----------|------|---------|-------------|----------------|
| `base_time` | int | 120 | Base time in minutes | ~2 hours for average user |
| `returning_user_reduction` | int | -30 | Time saved by experience | Familiar with process |
| `high_tech_reduction` | int | -20 | Time saved by tech skills | Navigate UI efficiently |
| `low_tech_penalty` | int | 40 | Extra time for tech struggles | UI confusion, errors |
| `std` | int | 30 | Time variability | Individual differences |
| `min_time` | int | 30 | Absolute minimum | Even experts need some time |

### Sessions

```yaml
sessions:
  base_sessions: 2.5          # Average number of sessions
  mobile_penalty: 0.5         # Extra sessions on mobile
  low_tech_penalty: 1.0       # Extra sessions for tech_savviness < 40
```

#### Session Parameters

| Parameter | Type | Default | Description | Business Logic |
|-----------|------|---------|-------------|----------------|
| `base_sessions` | float | 2.5 | Average sessions to complete | Most users need multiple sessions |
| `mobile_penalty` | float | 0.5 | Extra sessions on mobile | Smaller screen, interruptions |
| `low_tech_penalty` | float | 1.0 | Extra sessions for low-tech users | More confusion, restarts |

### Support Tickets

```yaml
support_tickets:
  base_rate: 0.15             # Base support ticket probability
  low_tech_boost: 0.15        # Extra support for tech_savviness < 30
  elderly_boost: 0.10         # Extra support for age > 65
```

#### Support Parameters

| Parameter | Type | Default | Description | Business Logic |
|-----------|------|---------|-------------|----------------|
| `base_rate` | float | 0.15 | Base support probability | ~15% of users need help |
| `low_tech_boost` | float | 0.15 | Extra support for low-tech | More confusion, errors |
| `elderly_boost` | float | 0.10 | Extra support for elderly | Technology barriers |

---

## Early Login Behavior

Controls early adoption indicators that predict treatment assignment.

```yaml
early_login:
  base_probability: 0.3       # Base early login probability
  high_tech_boost: 0.2        # Boost for tech_savviness > 60
  returning_user_boost: 0.15  # Boost for returning users
```

### Parameters

| Parameter | Type | Default | Description | Purpose |
|-----------|------|---------|-------------|---------|
| `base_probability` | float | 0.3 | Base early login rate | ~30% are early adopters |
| `high_tech_boost` | float | 0.2 | Boost for high-tech users | Tech enthusiasts login early |
| `returning_user_boost` | float | 0.15 | Boost for returning users | Familiar with platform timing |

### Business Logic
Early login behavior serves as a **predictor of treatment adoption**:
- Users who login early in tax season are more likely to try new features
- Creates realistic correlation between early adoption tendencies
- Helps model selection bias in feature adoption

---

## Treatment Assignment

Controls Smart Filing Assistant adoption probabilities and selection bias.

```yaml
treatment:
  base_adoption_rate: 0.4     # Base adoption probability (40%)
  
  tech_effects:
    high_threshold: 70        # Tech score for high boost
    high_boost: 0.25         # Adoption boost for high-tech users
    medium_threshold: 50      # Tech score for medium boost
    medium_boost: 0.10       # Adoption boost for medium-tech users
    low_threshold: 30         # Tech score for penalty
    low_penalty: -0.20       # Adoption penalty for low-tech users
  
  age_effects:
    young_threshold: 35       # Age for young boost
    young_boost: 0.15        # Adoption boost for young users
    old_threshold: 55         # Age for old penalty
    old_penalty: -0.10       # Adoption penalty for older users
  
  device_effects:
    mobile_boost: 0.05       # Small boost for mobile users
    tablet_penalty: -0.05    # Small penalty for tablet users
  
  early_login_boost: 0.20    # Strong boost for early login users
  returning_user_boost: 0.05 # Small boost for returning users
  
  income_effects:
    high_income_boost: 0.10  # Boost for >150k, 100k-150k income
    low_income_penalty: -0.05 # Penalty for <30k income
  
  min_probability: 0.05      # Minimum adoption probability
  max_probability: 0.95      # Maximum adoption probability
```

### Base Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_adoption_rate` | float | 0.4 | Starting adoption probability |
| `min_probability` | float | 0.05 | Floor for adoption probability |
| `max_probability` | float | 0.95 | Ceiling for adoption probability |

### Tech-Savviness Effects (Primary Driver)

| Parameter | Type | Default | Description | Rationale |
|-----------|------|---------|-------------|-----------|
| `high_threshold` | int | 70 | Tech score for high effect | Tech experts |
| `high_boost` | float | 0.25 | Strong boost for tech experts | Love trying new features |
| `medium_threshold` | int | 50 | Tech score for medium effect | Competent users |
| `medium_boost` | float | 0.10 | Moderate boost for competent users | Cautiously adopt |
| `low_threshold` | int | 30 | Tech score for penalty | Tech-struggling users |
| `low_penalty` | float | -0.20 | Penalty for low-tech users | Avoid new complexity |

### Age Effects

| Parameter | Type | Default | Description | Rationale |
|-----------|------|---------|-------------|-----------|
| `young_threshold` | int | 35 | Age cutoff for young boost | Digital natives |
| `young_boost` | float | 0.15 | Boost for young users | Comfortable with new tech |
| `old_threshold` | int | 55 | Age cutoff for old penalty | Traditional users |
| `old_penalty` | float | -0.10 | Penalty for older users | Prefer familiar interfaces |

### Other Effects

| Category | Parameter | Default | Rationale |
|----------|-----------|---------|-----------|
| **Device** | `mobile_boost` | 0.05 | Mobile users may appreciate assistance |
| **Device** | `tablet_penalty` | -0.05 | Tablet UX may be suboptimal |
| **Behavior** | `early_login_boost` | 0.20 | Early adopters try new features |
| **Experience** | `returning_user_boost` | 0.05 | Familiar users more willing to experiment |
| **Income** | `high_income_boost` | 0.10 | Higher income users try premium features |
| **Income** | `low_income_penalty` | -0.05 | Cost-conscious users stick to basics |

### Selection Bias Modeling

The treatment assignment creates **realistic selection bias**:
1. **Primary driver**: Tech-savviness (strongest effect)
2. **Secondary factors**: Age, early adoption behavior
3. **Tertiary factors**: Income, device type, user experience
4. **Cumulative effects**: Multiple factors combine additively
5. **Realistic bounds**: Probabilities capped at reasonable ranges

---

## Outcomes (2024)

Controls post-treatment outcomes and treatment effects.

### Filing Conversion

```yaml
outcomes_2024:
  filing:
    base_rate: 0.72           # Base 2024 filing rate (lower than 2023)
    
    treatment_effects:
      base_effect: 0.08       # Base treatment effect (8 percentage points)
      low_tech_boost: 0.05    # Extra benefit for tech_savviness < 40
      high_tech_reduction: -0.02 # Reduced benefit for tech_savviness > 70
      older_user_boost: 0.03  # Extra benefit for age > 55
      new_user_boost: 0.04    # Extra benefit for new users
    
    filed_2023_boost: 0.15    # Strong persistence from previous filing
    apply_demographic_effects: true # Apply same demographic effects as 2023
    max_probability: 0.95     # Maximum filing probability
```

#### Filing Parameters

| Parameter | Type | Default | Description | Business Logic |
|-----------|------|---------|-------------|----------------|
| `base_rate` | float | 0.72 | 2024 base filing rate | Slightly lower than 2023 |
| `base_effect` | float | 0.08 | Main treatment effect | 8pp increase from Smart Assistant |
| `filed_2023_boost` | float | 0.15 | Persistence effect | Strong habit formation |
| `max_probability` | float | 0.95 | Probability ceiling | Practical maximum |

#### Heterogeneous Treatment Effects

| Parameter | Type | Default | Description | Rationale |
|-----------|------|---------|-------------|-----------|
| `low_tech_boost` | float | 0.05 | Extra benefit for low-tech users | Assistance helps most where needed |
| `high_tech_reduction` | float | -0.02 | Reduced benefit for high-tech users | Already efficient, less room for improvement |
| `older_user_boost` | float | 0.03 | Extra benefit for older users | Guidance particularly valuable |
| `new_user_boost` | float | 0.04 | Extra benefit for new users | Hand-holding most valuable for newcomers |

### Time to Complete

```yaml
time_to_complete:
  treatment_time_reduction:
    min_reduction: 0.15       # Minimum time savings (15%)
    max_reduction: 0.25       # Maximum time savings (25%)
  std: 20                     # Standard deviation of completion times
  min_time: 20               # Minimum completion time
```

#### Time Parameters

| Parameter | Type | Default | Description | Business Impact |
|-----------|------|---------|-------------|-----------------|
| `min_reduction` | float | 0.15 | Minimum time savings | Conservative estimate |
| `max_reduction` | float | 0.25 | Maximum time savings | Optimistic but realistic |
| `std` | int | 20 | Time variability | Individual differences |
| `min_time` | int | 20 | Absolute minimum | Even with assistance |

### Sessions

```yaml
sessions:
  treatment_session_reduction:
    min_reduction: 0.10       # Minimum session reduction (10%)
    max_reduction: 0.30       # Maximum session reduction (30%)
```

#### Session Parameters

| Parameter | Type | Default | Description | Business Impact |
|-----------|------|---------|-------------|-----------------|
| `min_reduction` | float | 0.10 | Conservative session savings | Fewer interruptions/restarts |
| `max_reduction` | float | 0.30 | Optimistic session savings | Smoother completion flow |

### Support Tickets

```yaml
support_tickets:
  base_rate: 0.12             # Base 2024 support rate
  filed_2023_penalty: 0.05    # Extra support for previous filers with issues
  treatment_reduction: 0.6    # Support reduction factor (40% reduction)
```

#### Support Parameters

| Parameter | Type | Default | Description | Business Impact |
|-----------|------|---------|-------------|-----------------|
| `base_rate` | float | 0.12 | 2024 base support rate | Slightly improved from 2023 |
| `filed_2023_penalty` | float | 0.05 | Persistence of support needs | Previous issues predict future issues |
| `treatment_reduction` | float | 0.6 | Support reduction multiplier | Smart Assistant reduces support needs |

### User Satisfaction

```yaml
satisfaction:
  base_score: 7.2             # Base satisfaction (1-10 scale)
  treatment_boost: 0.8        # Satisfaction boost from treatment
  high_tech_boost: 0.3        # Extra satisfaction for tech_savviness > 60
  support_history_penalty: -0.5 # Satisfaction penalty for 2023 support tickets
  std: 1.2                    # Standard deviation of satisfaction scores
  min_score: 1               # Minimum satisfaction
  max_score: 10              # Maximum satisfaction
```

#### Satisfaction Parameters

| Parameter | Type | Default | Description | Business Impact |
|-----------|------|---------|-------------|-----------------|
| `base_score` | float | 7.2 | Baseline satisfaction | Above-average satisfaction |
| `treatment_boost` | float | 0.8 | Smart Assistant satisfaction boost | Users appreciate the help |
| `high_tech_boost` | float | 0.3 | Tech user satisfaction bonus | Appreciate sophisticated features |
| `support_history_penalty` | float | -0.5 | Penalty for past support issues | Previous frustrations linger |
| `std` | float | 1.2 | Satisfaction variability | Individual differences |

---

## Output Settings

Controls data output and derived feature generation.

```yaml
output:
  default_path: "data/simulated_users.csv"
  include_derived_features: true
  derived_features:
    - "time_improvement"      # 2023 time - 2024 time
    - "session_improvement"   # 2023 sessions - 2024 sessions
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_path` | str | "data/simulated_users.csv" | Default output file path |
| `include_derived_features` | bool | true | Whether to calculate derived features |
| `derived_features` | list[str] | [...] | List of derived features to calculate |

### Derived Features

| Feature | Formula | Description | Business Value |
|---------|---------|-------------|----------------|
| `time_improvement` | time_2023 - time_2024 | Time savings from treatment | Efficiency measurement |
| `session_improvement` | sessions_2023 - sessions_2024 | Session reduction from treatment | User experience improvement |

---

## Parameter Interactions

Understanding how parameters interact is crucial for realistic simulation.

### Primary Interaction Chains

1. **Tech-Savviness → Treatment → Outcomes**
   - High tech-savviness → Higher treatment adoption → Moderate treatment effects
   - Low tech-savviness → Lower treatment adoption → Strong treatment effects

2. **Age → Tech-Savviness → Treatment → Outcomes**
   - Young age → High tech-savviness → High treatment adoption
   - Old age → Low tech-savviness → Low treatment adoption → Strong effects when adopted

3. **Income → Tech-Savviness → Baseline Efficiency**
   - High income → Better tech access → Higher efficiency
   - Low income → Limited tech access → Lower efficiency

4. **User Type → Baseline Behavior → Treatment Effects**
   - Returning users → Established patterns → Moderate treatment effects
   - New users → No established patterns → Strong treatment effects

### Selection Bias Mechanisms

The simulation creates **realistic selection bias** through:

1. **Observed confounders**: Age, income, device type (measurable in real data)
2. **Unobserved confounders**: Tech-savviness (hidden but affects everything)
3. **Mediating variables**: Early login behavior (observable proxy for adoption tendency)
4. **Historical patterns**: 2023 behavior predicts 2024 behavior

### Heterogeneous Treatment Effects

Treatment effects vary systematically by:

1. **Tech level**: Assistance helps most where skills are lowest
2. **Experience**: New users benefit more from guidance
3. **Age**: Older users appreciate hand-holding
4. **Baseline efficiency**: More room for improvement = larger effects

---

## Creating Custom Scenarios

### Scenario Design Principles

1. **Start with business question**: What assumption do you want to test?
2. **Change minimal parameters**: Only override what's necessary
3. **Maintain realism**: Keep parameters within reasonable ranges
4. **Document rationale**: Explain why you changed specific parameters

### Common Scenario Types

#### 1. Treatment Effect Sensitivity

```yaml
# Test higher treatment effects
outcomes_2024:
  filing:
    treatment_effects:
      base_effect: 0.12       # 12% instead of 8%
```

#### 2. Adoption Rate Variations

```yaml
# Test lower adoption scenarios
treatment:
  base_adoption_rate: 0.25    # 25% instead of 40%
  tech_effects:
    high_boost: 0.20         # Reduced from 0.25
```

#### 3. Demographic Shifts

```yaml
# Test younger user base
demographics:
  age:
    mean: 35                 # Younger average age
    std: 12                  # Tighter distribution
```

#### 4. Tech-Savviness Distributions

```yaml
# Test higher baseline tech-savviness
tech_savviness:
  base_score: 60            # Higher baseline
  age_adjustments:
    old_penalty: -10         # Reduced age penalty
```

### Validation Checklist

When creating scenarios, verify:

- [ ] **Probabilities sum to 1.0** for categorical distributions
- [ ] **Realistic ranges** for all parameters
- [ ] **Logical consistency** between related parameters
- [ ] **Business rationale** for each change
- [ ] **Expected outcomes** align with scenario purpose

### Example: "Mobile-First" Scenario

**Note**: Thanks to configuration inheritance, you only need to override the parameters you want to change. The system automatically merges with the base configuration.

```yaml
# config/mobile_first_scenario.yaml
# Mobile-first scenario - only override what changes
simulation:
  random_seed: 789

demographics:
  device_types:
    weights: [0.70, 0.25, 0.05]  # Much higher mobile adoption
  age:
    mean: 35                     # Younger user base
    std: 12

treatment:
  device_effects:
    mobile_boost: 0.15          # Stronger mobile treatment adoption
    
outcomes_2024:
  filing:
    treatment_effects:
      base_effect: 0.10         # Stronger effects due to mobile optimization
```

**Important**: When creating scenario files, you only specify the parameters you want to change. All other parameters automatically inherit from `simulation_config.yaml`.

### Testing Scenarios

Always test new scenarios to ensure they work with inheritance:

```python
from src.data_simulation import TaxSoftwareDataSimulator

# Test the scenario
simulator = TaxSoftwareDataSimulator(
    n_users=1000,
    config_path="config/mobile_first_scenario.yaml"
)
df = simulator.generate_complete_dataset()

# Validate results match your expectations
print(f"Mobile usage: {(df['device_type'] == 'mobile').mean():.1%}")
print(f"Treatment rate: {df['used_smart_assistant'].mean():.1%}")
print(f"Average age: {df['age'].mean():.1f}")

# Verify inheritance worked by checking a non-overridden parameter
print(f"Base tech score: {simulator.config['tech_savviness']['base_score']}")
```

### Troubleshooting Scenarios

**Common Issues:**

1. **KeyError**: Missing required sections
   - **Solution**: Only override parameters you want to change, inheritance handles the rest

2. **Unexpected results**: Parameters not taking effect
   - **Solution**: Check parameter paths match the base config structure exactly

3. **Weights don't sum to 1.0**: Categorical distribution errors  
   - **Solution**: Ensure probability weights sum to 1.0 for categorical variables

---

## Appendix: Default Parameter Summary

### Quick Reference Table

| Category | Key Parameters | Default Values | Impact |
|----------|---------------|----------------|---------|
| **Demographics** | Age mean/std | 45 ± 15 | User base characteristics |
| **Tech-Savviness** | Base score | 50/100 | Primary confounder |
| **Treatment** | Base adoption | 40% | Selection bias strength |
| **Treatment Effects** | Base effect | 8pp | Treatment impact size |
| **Time Effects** | Reduction range | 15-25% | Efficiency improvements |
| **Support Effects** | Reduction factor | 40% | Support load reduction |

### Parameter Sensitivity

**High-impact parameters** (small changes have large effects):
- `treatment.base_adoption_rate`
- `outcomes_2024.filing.treatment_effects.base_effect`
- `tech_savviness.base_score`

**Medium-impact parameters** (moderate sensitivity):
- Age and income distributions
- Tech-savviness adjustments
- Heterogeneous effect modifiers

**Low-impact parameters** (fine-tuning):
- Standard deviations
- Minimum/maximum bounds
- Derived feature settings

---

*This documentation is maintained alongside the configuration system. When adding new parameters, please update this reference guide.* 