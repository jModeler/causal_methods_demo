# Configuration System Documentation

This directory contains YAML configuration files that control all parameters of the synthetic data simulation.

## 📁 Configuration Files

### `simulation_config.yaml` (Base Configuration)
The main configuration file containing all default parameters for data simulation:

- **Demographics**: Age, income, device, region distributions
- **Tech-savviness**: Scoring algorithm with age/income/region adjustments  
- **Baseline Behavior (2023)**: Filing rates, completion times, support tickets
- **Treatment Assignment**: Smart Assistant adoption probabilities
- **Outcomes (2024)**: Treatment effects and secondary metrics
- **Output Settings**: File paths and derived feature configuration

### `scenario_high_treatment.yaml` (High Effect Scenario)
Alternative scenario with stronger treatment effects:
- **Base treatment effect**: 15% (vs 8% in baseline)
- **Time reduction**: 25-40% (vs 15-25% in baseline)  
- **Support reduction**: 60% (vs 40% in baseline)
- **Satisfaction boost**: 1.2 points (vs 0.8 in baseline)

### `scenario_low_adoption.yaml` (Low Adoption Scenario)
Alternative scenario with lower treatment adoption rates:
- **Base adoption rate**: 25% (vs 40% in baseline)
- **Stronger penalties**: For low tech-savviness, older age
- **Compensating effects**: Slightly stronger treatment effects for those who adopt

## 🔧 How Configuration Inheritance Works

Scenario-specific configs inherit from the base configuration:

1. **Base config** (`simulation_config.yaml`) provides all default values
2. **Scenario configs** override only specific parameters they want to change
3. **Automatic merging** combines base + scenario parameters seamlessly

## 💡 Configuration Examples

### Changing Treatment Base Rate
```yaml
# In scenario file
treatment:
  base_adoption_rate: 0.3  # 30% instead of default 40%
```

### Modifying Treatment Effects
```yaml
# In scenario file  
outcomes_2024:
  filing:
    treatment_effects:
      base_effect: 0.12  # 12% instead of default 8%
      low_tech_boost: 0.06  # Stronger effect for low-tech users
```

### Adding New Demographics
```yaml
# In scenario file
demographics:
  income_brackets:
    values: ["<25k", "25k-50k", "50k-75k", "75k-100k", "100k+"]
    weights: [0.2, 0.25, 0.25, 0.2, 0.1]
```

## 🎯 Creating New Scenarios

1. **Create new YAML file** in this directory
2. **Override specific parameters** you want to change
3. **Leave unchanged parameters** out (they inherit from base)
4. **Use in code**:
   ```python
   from src.data_simulation import TaxSoftwareDataSimulator
   simulator = TaxSoftwareDataSimulator(
       config_path="config/my_scenario.yaml"
   )
   ```

## 📊 Parameter Categories

### Demographics
- Age distribution (mean, std, min/max)
- Income bracket values and weights
- Device type preferences
- Geographic region distributions

### Tech-Savviness Scoring
- Base score and variance
- Age-based adjustments
- Income-based boosts/penalties
- Regional effects

### Treatment Assignment
- Base adoption probability
- Tech-savviness thresholds and effects
- Age, device, income modifiers
- Bounds and constraints

### Baseline Behavior (2023)
- Filing probability factors
- Time to complete parameters
- Session count distributions
- Support ticket rates

### Outcomes (2024)
- Treatment effect sizes
- Heterogeneous effect modifiers
- Secondary outcome parameters
- Satisfaction scoring

## 🧪 Testing Scenarios

Use this script to test all scenarios:

```python
from src.data_simulation import TaxSoftwareDataSimulator
import os

scenarios = [
    'simulation_config.yaml',
    'scenario_high_treatment.yaml', 
    'scenario_low_adoption.yaml'
]

for scenario in scenarios:
    config_path = f'config/{scenario}'
    simulator = TaxSoftwareDataSimulator(
        n_users=1000, 
        config_path=config_path
    )
    df = simulator.generate_complete_dataset()
    print(f"{scenario}: {df['used_smart_assistant'].mean():.1%} treatment rate")
```

## 🔄 Version Control

- **Track config changes** in git for reproducibility
- **Document parameter rationale** in commit messages
- **Tag scenario versions** for important analyses
- **Share configs** with team for consistent results 