# Test Information Criteria in DML

from src.causal_methods.dml import DoubleMachineLearning
from src.data_simulation import generate_and_save_data
import pandas as pd

# Generate test data
df = generate_and_save_data(output_path=None, n_users=500)
dml = DoubleMachineLearning(df, random_state=42)

# Compare models using information criteria
print("🔍 Comparing Models with Information Criteria:")
print("=" * 60)

comparison = dml.compare_models(
    outcome_col="filed_2024",
    treatment_col="used_smart_assistant", 
    covariates=["age", "tech_savviness", "filed_2023"],
    n_folds=2
)

# Show model comparison with AIC/BIC
if "aic" in comparison.columns and "bic" in comparison.columns:
    print("\\nModel Comparison with Information Criteria:")
    display_cols = ["outcome_model", "treatment_model", "ate", "aic", "bic"]
    available_cols = [col for col in display_cols if col in comparison.columns]
    print(comparison[available_cols].round(4))
    
    print("\\n📊 Model Selection Guidelines:")
    print("• Lower AIC/BIC values indicate better model fit")
    print("• AIC: Akaike Information Criterion (balance fit vs complexity)")
    print("• BIC: Bayesian Information Criterion (stronger penalty for complexity)")
    
    best_aic_idx = comparison["aic"].idxmin()
    best_bic_idx = comparison["bic"].idxmin()
    
    print(f"\\n🏆 Best Models:")
    print(f"• Best AIC: {comparison.loc[best_aic_idx, \"outcome_model\"]} + {comparison.loc[best_aic_idx, \"treatment_model\"]} (AIC: {comparison.loc[best_aic_idx, \"aic\"]:.2f})")
    print(f"• Best BIC: {comparison.loc[best_bic_idx, \"outcome_model\"]} + {comparison.loc[best_bic_idx, \"treatment_model\"]} (BIC: {comparison.loc[best_bic_idx, \"bic\"]:.2f})")
else:
    print("Information criteria not available in comparison results")

