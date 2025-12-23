"""
Synthetic Control Method for causal inference.

This module implements synthetic control methodology for estimating treatment effects
by constructing synthetic control units from weighted combinations of donor units.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


class SyntheticControl:
    """
    Synthetic Control Method for causal inference.
    
    This implementation focuses on individual-level synthetic control where each
    treated unit gets its own synthetic control constructed from donor units.
    """

    def __init__(self, data: pd.DataFrame, random_state: int | None = None):
        """
        Initialize Synthetic Control analyzer.
        
        Args:
            data: DataFrame with units, treatments, and outcomes
            random_state: Random seed for reproducibility
        """
        self.data = data.copy()
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.synthetic_weights = {}
        self.results = {}
        self.donor_pool = None
        self.treated_units = None

    def prepare_data(
        self,
        unit_id_col: str = 'user_id',
        treatment_col: str = 'used_smart_assistant',
        outcome_pre_col: str = 'filed_2023',
        outcome_post_col: str = 'filed_2024',
        predictor_cols: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Prepare data for synthetic control analysis.
        
        Args:
            unit_id_col: Column identifying units
            treatment_col: Binary treatment indicator
            outcome_pre_col: Pre-treatment outcome
            outcome_post_col: Post-treatment outcome
            predictor_cols: Predictor variables for matching
            
        Returns:
            Dictionary with prepared data components
        """
        if predictor_cols is None:
            # Default predictors for tax software data
            predictor_cols = [
                'filed_2023', 'time_to_complete_2023', 'sessions_2023',
                'support_tickets_2023', 'age', 'tech_savviness'
            ]

        # Filter to available predictors
        available_predictors = [col for col in predictor_cols if col in self.data.columns]

        if not available_predictors:
            raise ValueError("No predictor columns found in data")

        # Split into treated and control (donor pool)
        treated_mask = self.data[treatment_col] == 1
        control_mask = self.data[treatment_col] == 0

        self.treated_units = self.data[treated_mask].copy()
        self.donor_pool = self.data[control_mask].copy()

        if len(self.treated_units) == 0:
            raise ValueError("No treated units found")
        if len(self.donor_pool) == 0:
            raise ValueError("No control units found for donor pool")

        # Prepare predictor matrices
        X_treated = self.treated_units[available_predictors].values
        X_donors = self.donor_pool[available_predictors].values

        # Convert boolean columns to float for consistency
        X_treated = X_treated.astype(float)
        X_donors = X_donors.astype(float)

        # Handle missing values
        if np.any(np.isnan(X_treated)) or np.any(np.isnan(X_donors)):
            warnings.warn("Missing values detected. Using mean imputation.")
            # Simple mean imputation
            all_X = np.vstack([X_treated, X_donors])
            means = np.nanmean(all_X, axis=0)

            for i in range(X_treated.shape[1]):
                X_treated[np.isnan(X_treated[:, i]), i] = means[i]
                X_donors[np.isnan(X_donors[:, i]), i] = means[i]

        return {
            'X_treated': X_treated,
            'X_donors': X_donors,
            'predictor_cols': available_predictors,
            'unit_id_col': unit_id_col,
            'treatment_col': treatment_col,
            'outcome_pre_col': outcome_pre_col,
            'outcome_post_col': outcome_post_col,
            'n_treated': len(self.treated_units),
            'n_donors': len(self.donor_pool)
        }

    def _optimize_weights(
        self,
        treated_characteristics: np.ndarray,
        donor_characteristics: np.ndarray,
        method: str = 'minimize_distance'
    ) -> np.ndarray:
        """
        Optimize synthetic control weights for a single treated unit.
        
        Args:
            treated_characteristics: Characteristics of treated unit
            donor_characteristics: Characteristics of donor units
            method: Optimization method
            
        Returns:
            Optimal weights for donor units
        """
        n_donors = donor_characteristics.shape[0]

        # Objective function: minimize distance between treated and synthetic control
        def objective(weights):
            synthetic_char = np.dot(weights, donor_characteristics)
            return np.sum((treated_characteristics - synthetic_char) ** 2)

        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_donors)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_donors) / n_donors

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
            return initial_weights

        return result.x

    def construct_synthetic_controls(
        self,
        unit_id_col: str = 'user_id',
        treatment_col: str = 'used_smart_assistant',
        outcome_pre_col: str = 'filed_2023',
        outcome_post_col: str = 'filed_2024',
        predictor_cols: list[str] | None = None,
        standardize: bool = True
    ) -> dict[str, Any]:
        """
        Construct synthetic controls for all treated units.
        
        Args:
            unit_id_col: Column identifying units
            treatment_col: Binary treatment indicator
            outcome_pre_col: Pre-treatment outcome
            outcome_post_col: Post-treatment outcome  
            predictor_cols: Predictor variables for matching
            standardize: Whether to standardize predictors
            
        Returns:
            Dictionary with synthetic control results
        """
        # Prepare data
        data_prep = self.prepare_data(
            unit_id_col, treatment_col, outcome_pre_col,
            outcome_post_col, predictor_cols
        )

        X_treated = data_prep['X_treated']
        X_donors = data_prep['X_donors']

        # Standardize predictors if requested
        if standardize:
            scaler = StandardScaler()
            X_combined = np.vstack([X_treated, X_donors])
            X_combined_scaled = scaler.fit_transform(X_combined)

            X_treated = X_combined_scaled[:len(X_treated)]
            X_donors = X_combined_scaled[len(X_treated):]

        # Construct synthetic controls
        results = []
        weights_dict = {}

        for i, treated_id in enumerate(self.treated_units[unit_id_col]):
            # Get optimal weights for this treated unit
            weights = self._optimize_weights(X_treated[i], X_donors)
            weights_dict[treated_id] = weights

            # Calculate synthetic control outcomes
            donor_pre_outcomes = self.donor_pool[outcome_pre_col].values.astype(float)
            donor_post_outcomes = self.donor_pool[outcome_post_col].values.astype(float)

            synthetic_pre = np.dot(weights, donor_pre_outcomes)
            synthetic_post = np.dot(weights, donor_post_outcomes)

            # Get actual outcomes
            actual_pre = float(self.treated_units.iloc[i][outcome_pre_col])
            actual_post = float(self.treated_units.iloc[i][outcome_post_col])

            # Calculate treatment effect
            actual_change = actual_post - actual_pre
            synthetic_change = synthetic_post - synthetic_pre
            individual_effect = actual_change - synthetic_change

            # Calculate pre-treatment fit quality
            pre_treatment_error = abs(actual_pre - synthetic_pre)

            results.append({
                'unit_id': treated_id,
                'actual_pre': actual_pre,
                'actual_post': actual_post,
                'synthetic_pre': synthetic_pre,
                'synthetic_post': synthetic_post,
                'actual_change': actual_change,
                'synthetic_change': synthetic_change,
                'treatment_effect': individual_effect,
                'pre_treatment_error': pre_treatment_error,
                'weights_concentration': np.sum(weights ** 2)  # Measure of weight concentration
            })

        self.synthetic_weights = weights_dict
        results_df = pd.DataFrame(results)

        # Calculate aggregate statistics
        aggregate_results = {
            'individual_results': results_df,
            'average_treatment_effect': results_df['treatment_effect'].mean(),
            'ate_std_error': results_df['treatment_effect'].std() / np.sqrt(len(results_df)),
            'median_treatment_effect': results_df['treatment_effect'].median(),
            'treatment_effect_distribution': results_df['treatment_effect'].describe(),
            'average_pre_treatment_error': results_df['pre_treatment_error'].mean(),
            'weight_concentration': results_df['weights_concentration'].mean(),
            'data_preparation': data_prep
        }

        self.results = aggregate_results
        return aggregate_results

    def estimate_statistical_significance(self, n_placebo: int = 100) -> dict[str, Any]:
        """
        Estimate statistical significance using placebo tests.
        
        Args:
            n_placebo: Number of placebo tests to run
            
        Returns:
            Dictionary with significance test results
        """
        if not self.results:
            raise ValueError("Must run construct_synthetic_controls() first")

        observed_ate = self.results['average_treatment_effect']
        placebo_effects = []

        # Get data preparation info
        data_prep = self.results['data_preparation']

        # Run placebo tests
        for _ in range(n_placebo):
            # Randomly assign treatment among all units
            all_units = self.data.copy()
            n_treated = len(self.treated_units)

            # Random treatment assignment
            treated_indices = np.random.choice(
                len(all_units), size=n_treated, replace=False
            )
            placebo_treatment = np.zeros(len(all_units))
            placebo_treatment[treated_indices] = 1
            all_units['placebo_treatment'] = placebo_treatment

            # Create temporary synthetic control object
            temp_sc = SyntheticControl(all_units, random_state=None)

            try:
                placebo_results = temp_sc.construct_synthetic_controls(
                    treatment_col='placebo_treatment',
                    **{k: v for k, v in data_prep.items()
                       if k in ['unit_id_col', 'outcome_pre_col', 'outcome_post_col']}
                )
                placebo_effects.append(placebo_results['average_treatment_effect'])
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                # Skip failed placebo tests
                continue

        if len(placebo_effects) == 0:
            warnings.warn("All placebo tests failed")
            return {'p_value': np.nan, 'placebo_effects': []}

        # Calculate p-value
        placebo_effects = np.array(placebo_effects)
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(observed_ate))

        return {
            'observed_ate': observed_ate,
            'placebo_effects': placebo_effects,
            'p_value': p_value,
            'n_placebo_tests': len(placebo_effects),
            'placebo_mean': np.mean(placebo_effects),
            'placebo_std': np.std(placebo_effects)
        }

    def plot_treatment_effects(self, figsize: tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive visualization of synthetic control results.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("Must run construct_synthetic_controls() first")

        results_df = self.results['individual_results']

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Synthetic Control Analysis Results', fontsize=16, fontweight='bold')

        # Plot 1: Individual treatment effects
        axes[0, 0].hist(results_df['treatment_effect'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.results['average_treatment_effect'], color='red', linestyle='--',
                          label=f'ATE: {self.results["average_treatment_effect"]:.3f}')
        axes[0, 0].set_title('Distribution of Individual Treatment Effects')
        axes[0, 0].set_xlabel('Treatment Effect')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Pre-treatment fit quality
        axes[0, 1].scatter(results_df['actual_pre'], results_df['synthetic_pre'], alpha=0.6)
        min_val = min(results_df['actual_pre'].min(), results_df['synthetic_pre'].min())
        max_val = max(results_df['actual_pre'].max(), results_df['synthetic_pre'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        axes[0, 1].set_title('Pre-treatment Fit Quality')
        axes[0, 1].set_xlabel('Actual Pre-treatment Outcome')
        axes[0, 1].set_ylabel('Synthetic Pre-treatment Outcome')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Actual vs Synthetic trends
        outcomes_actual = np.column_stack([results_df['actual_pre'], results_df['actual_post']])
        outcomes_synthetic = np.column_stack([results_df['synthetic_pre'], results_df['synthetic_post']])

        mean_actual = np.mean(outcomes_actual, axis=0)
        mean_synthetic = np.mean(outcomes_synthetic, axis=0)

        periods = ['Pre-treatment', 'Post-treatment']
        axes[0, 2].plot(periods, mean_actual, 'bo-', label='Actual (Treated)', linewidth=2, markersize=8)
        axes[0, 2].plot(periods, mean_synthetic, 'ro-', label='Synthetic Control', linewidth=2, markersize=8)
        axes[0, 2].set_title('Average Outcomes: Treated vs Synthetic')
        axes[0, 2].set_ylabel('Outcome Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Treatment effect vs pre-treatment error
        axes[1, 0].scatter(results_df['pre_treatment_error'], results_df['treatment_effect'], alpha=0.6)
        axes[1, 0].set_title('Treatment Effect vs Pre-treatment Fit')
        axes[1, 0].set_xlabel('Pre-treatment Error')
        axes[1, 0].set_ylabel('Treatment Effect')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Weight concentration
        axes[1, 1].hist(results_df['weights_concentration'], bins=15, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Synthetic Control Weight Concentration')
        axes[1, 1].set_xlabel('Sum of Squared Weights')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Individual outcomes comparison
        x_pos = np.arange(len(results_df))
        width = 0.35

        axes[1, 2].bar(x_pos - width/2, results_df['actual_change'], width,
                      label='Actual Change', alpha=0.7, color='blue')
        axes[1, 2].bar(x_pos + width/2, results_df['synthetic_change'], width,
                      label='Synthetic Change', alpha=0.7, color='red')
        axes[1, 2].set_title('Individual Outcome Changes')
        axes[1, 2].set_xlabel('Unit Index')
        axes[1, 2].set_ylabel('Change in Outcome')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Only show every 10th x-label to avoid crowding
        if len(results_df) > 20:
            tick_positions = x_pos[::max(1, len(x_pos)//10)]
            axes[1, 2].set_xticks(tick_positions)
            axes[1, 2].set_xticklabels([f'{i}' for i in tick_positions])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        return fig

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of synthetic control analysis.
        
        Returns:
            Formatted string report
        """
        if not self.results:
            return "No synthetic control results available. Run construct_synthetic_controls() first."

        results_df = self.results['individual_results']
        ate = self.results['average_treatment_effect']
        se = self.results['ate_std_error']

        report = "=" * 70 + "\n"
        report += "SYNTHETIC CONTROL ANALYSIS SUMMARY\n"
        report += "=" * 70 + "\n\n"

        # Basic statistics
        report += "1. TREATMENT EFFECT ESTIMATES:\n"
        report += "-" * 35 + "\n"
        report += f"Average Treatment Effect (ATE): {ate:.4f}\n"
        report += f"Standard Error: {se:.4f}\n"
        report += f"95% Confidence Interval: [{ate - 1.96*se:.4f}, {ate + 1.96*se:.4f}]\n"
        report += f"Median Treatment Effect: {self.results['median_treatment_effect']:.4f}\n\n"

        # Distribution statistics
        report += "2. TREATMENT EFFECT DISTRIBUTION:\n"
        report += "-" * 40 + "\n"
        te_stats = self.results['treatment_effect_distribution']
        report += f"Count: {te_stats['count']:.0f}\n"
        report += f"Mean: {te_stats['mean']:.4f}\n"
        report += f"Std: {te_stats['std']:.4f}\n"
        report += f"Min: {te_stats['min']:.4f}\n"
        report += f"25%: {te_stats['25%']:.4f}\n"
        report += f"50%: {te_stats['50%']:.4f}\n"
        report += f"75%: {te_stats['75%']:.4f}\n"
        report += f"Max: {te_stats['max']:.4f}\n\n"

        # Quality metrics
        report += "3. SYNTHETIC CONTROL QUALITY:\n"
        report += "-" * 35 + "\n"
        report += f"Average Pre-treatment Error: {self.results['average_pre_treatment_error']:.4f}\n"
        report += f"Average Weight Concentration: {self.results['weight_concentration']:.4f}\n"

        # Interpretation
        good_fit = self.results['average_pre_treatment_error'] < 0.1
        concentrated_weights = self.results['weight_concentration'] > 0.5

        report += f"Pre-treatment Fit Quality: {' Good' if good_fit else '️  Fair'}\n"
        report += f"Weight Concentration: {'️  High' if concentrated_weights else ' Dispersed'}\n\n"

        # Effect interpretation
        report += "4. EFFECT INTERPRETATION:\n"
        report += "-" * 30 + "\n"

        significant = abs(ate) > 2 * se
        effect_size = "Large" if abs(ate) > 0.1 else "Medium" if abs(ate) > 0.05 else "Small"
        direction = "Positive" if ate > 0 else "Negative" if ate < 0 else "Null"

        report += f"Effect Direction: {direction}\n"
        report += f"Effect Size: {effect_size}\n"
        report += f"Statistical Significance: {' Likely' if significant else ' Unlikely'}\n\n"

        # Recommendations
        report += "5. RECOMMENDATIONS:\n"
        report += "-" * 25 + "\n"

        if good_fit and not concentrated_weights:
            report += " High-quality synthetic controls - results are reliable\n"
        elif not good_fit:
            report += "️  Poor pre-treatment fit - consider more predictors\n"
        elif concentrated_weights:
            report += "️  Concentrated weights - results may be sensitive to outliers\n"

        if significant:
            report += " Consider implementing intervention based on positive results\n"
        else:
            report += "️  Inconclusive results - consider larger sample or longer observation period\n"

        report += "\n" + "=" * 70

        return report


def load_and_analyze_synthetic_control(
    file_path: str,
    unit_id_col: str = 'user_id',
    treatment_col: str = 'used_smart_assistant',
    outcome_pre_col: str = 'filed_2023',
    outcome_post_col: str = 'filed_2024',
    predictor_cols: list[str] | None = None
) -> dict[str, Any]:
    """
    Convenience function to load data and run synthetic control analysis.
    
    Args:
        file_path: Path to data file
        unit_id_col: Column identifying units
        treatment_col: Binary treatment indicator
        outcome_pre_col: Pre-treatment outcome
        outcome_post_col: Post-treatment outcome
        predictor_cols: Predictor variables for matching
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    # Run analysis
    sc = SyntheticControl(data, random_state=42)
    results = sc.construct_synthetic_controls(
        unit_id_col=unit_id_col,
        treatment_col=treatment_col,
        outcome_pre_col=outcome_pre_col,
        outcome_post_col=outcome_post_col,
        predictor_cols=predictor_cols
    )

    return {
        'synthetic_control': sc,
        'results': results,
        'summary_report': sc.generate_summary_report()
    }
