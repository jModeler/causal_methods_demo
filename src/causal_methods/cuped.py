"""
CUPED (Controlled-experiment Using Pre-Experiment Data) Implementation

CUPED is a variance reduction technique that uses pre-experiment covariates to reduce 
the variance of treatment effect estimates while preserving unbiasedness.

Key advantages:
- Reduces variance of treatment effect estimates
- Increases statistical power
- Works well with randomized experiments
- Preserves unbiasedness of estimates

Reference: Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). 
Improving the sensitivity of online controlled experiments by utilizing pre-experiment data.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class CUPED:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) for variance reduction.
    
    CUPED reduces the variance of treatment effect estimates by using pre-experiment
    covariates that are correlated with the outcome but uncorrelated with treatment
    assignment (in randomized experiments).
    """

    def __init__(self, data: pd.DataFrame, random_state: int | None = None):
        """
        Initialize CUPED analyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset containing outcomes, treatment, and pre-experiment covariates
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.data = data.copy()
        self.random_state = random_state
        self.results = {}
        self.cuped_adjustments = {}

        if random_state is not None:
            np.random.seed(random_state)

    def estimate_cuped_adjustment(
        self,
        outcome_col: str,
        covariate_cols: list[str],
        treatment_col: str | None = None,
        method: str = "ols"
    ) -> dict:
        """
        Estimate CUPED adjustment coefficients.
        
        Parameters:
        -----------
        outcome_col : str
            Name of outcome variable
        covariate_cols : List[str]
            Names of pre-experiment covariates
        treatment_col : Optional[str]
            Name of treatment variable (to check balance)
        method : str
            Method for estimating adjustment ('ols', 'ridge', 'lasso')
            
        Returns:
        --------
        Dict
            Adjustment coefficients and diagnostics
        """
        # Prepare data
        outcome = self.data[outcome_col].dropna()
        covariates = self.data[covariate_cols].loc[outcome.index]

        # Check for missing values
        if covariates.isnull().any().any():
            warnings.warn("Missing values in covariates. Consider imputation.")
            # Simple imputation with mean
            covariates = covariates.fillna(covariates.mean())

        # Estimate adjustment coefficients using control group or full sample
        if treatment_col is not None and treatment_col in self.data.columns:
            treatment = self.data[treatment_col].loc[outcome.index]
            # Use control group for estimation (standard CUPED practice)
            control_mask = (treatment == 0)
            if control_mask.sum() > len(covariate_cols) + 10:  # Ensure sufficient sample
                outcome_est = outcome[control_mask]
                covariates_est = covariates[control_mask]
            else:
                warnings.warn("Insufficient control group size. Using full sample.")
                outcome_est = outcome
                covariates_est = covariates
        else:
            outcome_est = outcome
            covariates_est = covariates

        # Fit regression model
        if method == "ols":
            model = LinearRegression()
        elif method == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
        elif method == "lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown method: {method}")

        model.fit(covariates_est, outcome_est)

        # Calculate adjustment coefficients (theta)
        theta = model.coef_ if hasattr(model, 'coef_') else np.array([0] * len(covariate_cols))
        intercept = model.intercept_ if hasattr(model, 'intercept_') else 0

        # Predict outcomes for variance calculation
        y_pred = model.predict(covariates_est)

        # Calculate R² and variance reduction
        r2 = r2_score(outcome_est, y_pred)
        variance_reduction = r2  # In CUPED, variance reduction ≈ R²

        # Calculate covariate means (for adjustment)
        covariate_means = covariates.mean()

        # Store adjustment info
        adjustment_info = {
            'theta': theta,
            'intercept': intercept,
            'covariate_means': covariate_means,
            'covariate_cols': covariate_cols,
            'method': method,
            'r2': r2,
            'variance_reduction': variance_reduction,
            'model': model,
            'n_estimation': len(outcome_est)
        }

        # Check covariate balance if treatment provided
        if treatment_col is not None:
            balance_check = self._check_covariate_balance(
                covariates, self.data[treatment_col].loc[outcome.index]
            )
            adjustment_info['balance_check'] = balance_check

        self.cuped_adjustments[outcome_col] = adjustment_info
        return adjustment_info

    def apply_cuped_adjustment(
        self,
        outcome_col: str,
        adjustment_info: dict | None = None
    ) -> pd.Series:
        """
        Apply CUPED adjustment to outcome variable.
        
        Parameters:
        -----------
        outcome_col : str
            Name of outcome variable
        adjustment_info : Optional[Dict]
            Pre-computed adjustment info (if None, uses stored adjustment)
            
        Returns:
        --------
        pd.Series
            CUPED-adjusted outcome variable
        """
        if adjustment_info is None:
            if outcome_col not in self.cuped_adjustments:
                raise ValueError(f"No adjustment computed for {outcome_col}. Run estimate_cuped_adjustment first.")
            adjustment_info = self.cuped_adjustments[outcome_col]

        # Get original outcome
        outcome = self.data[outcome_col]

        # Get covariates
        covariate_cols = adjustment_info['covariate_cols']
        covariates = self.data[covariate_cols]

        # Handle missing values
        valid_mask = outcome.notna() & covariates.notna().all(axis=1)

        # Apply CUPED formula: Y_cuped = Y - theta * (X - E[X])
        theta = adjustment_info['theta']
        covariate_means = adjustment_info['covariate_means']

        # Calculate covariate deviations
        covariate_deviations = covariates - covariate_means

        # Apply adjustment
        adjustment = np.dot(covariate_deviations, theta)
        cuped_outcome = outcome - adjustment

        # Set invalid values to NaN
        cuped_outcome[~valid_mask] = np.nan

        return cuped_outcome

    def estimate_treatment_effects(
        self,
        outcome_col: str,
        treatment_col: str,
        covariate_cols: list[str],
        adjustment_method: str = "ols",
        confidence_level: float = 0.95
    ) -> dict:
        """
        Estimate treatment effects with and without CUPED adjustment.
        
        Parameters:
        -----------
        outcome_col : str
            Name of outcome variable
        treatment_col : str
            Name of treatment variable
        covariate_cols : List[str]
            Names of pre-experiment covariates
        adjustment_method : str
            Method for CUPED adjustment
        confidence_level : float
            Confidence level for intervals
            
        Returns:
        --------
        Dict
            Treatment effect estimates with and without CUPED
        """
        # Estimate CUPED adjustment
        adjustment_info = self.estimate_cuped_adjustment(
            outcome_col, covariate_cols, treatment_col, adjustment_method
        )

        # Apply CUPED adjustment
        cuped_outcome = self.apply_cuped_adjustment(outcome_col, adjustment_info)

        # Get treatment assignment
        treatment = self.data[treatment_col]

        # Create analysis dataset
        analysis_data = pd.DataFrame({
            'outcome_original': self.data[outcome_col],
            'outcome_cuped': cuped_outcome,
            'treatment': treatment
        }).dropna()

        # Estimate treatment effects
        results = {}

        # Original (unadjusted) analysis
        original_results = self._estimate_simple_treatment_effect(
            analysis_data['outcome_original'],
            analysis_data['treatment'],
            confidence_level
        )
        original_results['method'] = 'Original (No CUPED)'
        results['original'] = original_results

        # CUPED-adjusted analysis
        cuped_results = self._estimate_simple_treatment_effect(
            analysis_data['outcome_cuped'],
            analysis_data['treatment'],
            confidence_level
        )
        cuped_results['method'] = 'CUPED-Adjusted'
        results['cuped'] = cuped_results

        # Calculate variance reduction
        variance_reduction = 1 - (cuped_results['se'] ** 2) / (original_results['se'] ** 2)

        # Calculate statistical power improvement
        power_improvement = original_results['se'] / cuped_results['se']

        # Summary statistics
        results['summary'] = {
            'variance_reduction': variance_reduction,
            'power_improvement': power_improvement,
            'se_reduction': 1 - cuped_results['se'] / original_results['se'],
            'adjustment_r2': adjustment_info['r2'],
            'n_samples': len(analysis_data),
            'covariate_cols': covariate_cols
        }

        # Store results
        self.results[outcome_col] = results

        return results

    def _estimate_simple_treatment_effect(
        self,
        outcome: pd.Series,
        treatment: pd.Series,
        confidence_level: float = 0.95
    ) -> dict:
        """
        Estimate simple treatment effect (difference in means).
        
        Parameters:
        -----------
        outcome : pd.Series
            Outcome variable
        treatment : pd.Series
            Treatment assignment (0/1)
        confidence_level : float
            Confidence level
            
        Returns:
        --------
        Dict
            Treatment effect estimates
        """
        # Split by treatment
        treated = outcome[treatment == 1]
        control = outcome[treatment == 0]

        # Calculate means
        mean_treated = treated.mean()
        mean_control = control.mean()
        ate = mean_treated - mean_control

        # Calculate standard error
        var_treated = treated.var(ddof=1)
        var_control = control.var(ddof=1)
        n_treated = len(treated)
        n_control = len(control)

        se = np.sqrt(var_treated / n_treated + var_control / n_control)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        df = n_treated + n_control - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)

        ci_lower = ate - t_critical * se
        ci_upper = ate + t_critical * se

        # Calculate p-value
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            't_stat': t_stat,
            'n_treated': n_treated,
            'n_control': n_control,
            'mean_treated': mean_treated,
            'mean_control': mean_control
        }

    def _check_covariate_balance(
        self,
        covariates: pd.DataFrame,
        treatment: pd.Series
    ) -> dict:
        """
        Check balance of covariates between treatment and control groups.
        
        Parameters:
        -----------
        covariates : pd.DataFrame
            Covariate values
        treatment : pd.Series
            Treatment assignment
            
        Returns:
        --------
        Dict
            Balance check results
        """
        balance_results = {}

        for col in covariates.columns:
            covariate = covariates[col]
            treated_values = covariate[treatment == 1]
            control_values = covariate[treatment == 0]

            # Calculate standardized difference
            mean_diff = treated_values.mean() - control_values.mean()
            pooled_std = np.sqrt((treated_values.var() + control_values.var()) / 2)
            std_diff = mean_diff / pooled_std if pooled_std > 0 else 0

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(treated_values, control_values, equal_var=False)

            balance_results[col] = {
                'mean_treated': treated_values.mean(),
                'mean_control': control_values.mean(),
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'balanced': abs(std_diff) < 0.1  # Common threshold
            }

        return balance_results

    def plot_cuped_comparison(
        self,
        outcome_col: str,
        treatment_col: str,
        figsize: tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot comparison of original vs CUPED-adjusted analysis.
        
        Parameters:
        -----------
        outcome_col : str
            Name of outcome variable
        treatment_col : str
            Name of treatment variable
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure with comparison plots
        """
        if outcome_col not in self.results:
            raise ValueError(f"No results found for {outcome_col}. Run estimate_treatment_effects first.")

        results = self.results[outcome_col]

        # Get adjusted outcome
        cuped_outcome = self.apply_cuped_adjustment(outcome_col)
        treatment = self.data[treatment_col]

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'CUPED Analysis: {outcome_col}', fontsize=16, fontweight='bold')

        # Plot 1: Original outcome distributions
        treated_orig = self.data[outcome_col][treatment == 1]
        control_orig = self.data[outcome_col][treatment == 0]

        # Convert boolean outcomes to integers for plotting
        if treated_orig.dtype == 'bool' or control_orig.dtype == 'bool':
            treated_orig = treated_orig.astype(int)
            control_orig = control_orig.astype(int)
            # Use fewer bins for binary outcomes
            hist_bins = [0, 0.5, 1]
        else:
            hist_bins = 30

        axes[0, 0].hist(control_orig, alpha=0.6, label='Control', bins=hist_bins, color='red')
        axes[0, 0].hist(treated_orig, alpha=0.6, label='Treated', bins=hist_bins, color='blue')
        axes[0, 0].set_title('Original Outcome Distribution')
        axes[0, 0].set_xlabel('Outcome Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # Plot 2: CUPED-adjusted outcome distributions
        treated_cuped = cuped_outcome[treatment == 1]
        control_cuped = cuped_outcome[treatment == 0]

        # Handle CUPED outcomes (these are continuous even if original was binary)
        axes[0, 1].hist(control_cuped.dropna(), alpha=0.6, label='Control', bins=30, color='red')
        axes[0, 1].hist(treated_cuped.dropna(), alpha=0.6, label='Treated', bins=30, color='blue')
        axes[0, 1].set_title('CUPED-Adjusted Outcome Distribution')
        axes[0, 1].set_xlabel('Adjusted Outcome Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Plot 3: Treatment effect comparison
        methods = ['Original', 'CUPED']
        ates = [results['original']['ate'], results['cuped']['ate']]
        ses = [results['original']['se'], results['cuped']['se']]

        x_pos = np.arange(len(methods))
        axes[0, 2].bar(x_pos, ates, yerr=ses, capsize=5, alpha=0.7, color=['orange', 'green'])
        axes[0, 2].set_xlabel('Method')
        axes[0, 2].set_ylabel('Average Treatment Effect')
        axes[0, 2].set_title('Treatment Effect Comparison')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(methods)
        axes[0, 2].grid(True, alpha=0.3)

        # Add ATE values as text
        for i, (ate, se) in enumerate(zip(ates, ses, strict=False)):
            axes[0, 2].text(i, ate + se + 0.01 * max(ates), f'{ate:.4f}',
                           ha='center', va='bottom', fontweight='bold')

        # Plot 4: Variance comparison
        variances = [results['original']['se']**2, results['cuped']['se']**2]
        axes[1, 0].bar(x_pos, variances, alpha=0.7, color=['orange', 'green'])
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Variance of Treatment Effect')
        axes[1, 0].set_title('Variance Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(methods)
        axes[1, 0].grid(True, alpha=0.3)

        # Add variance reduction text
        var_reduction = results['summary']['variance_reduction']
        axes[1, 0].text(0.5, max(variances) * 0.8, f'Variance Reduction:\n{var_reduction:.1%}',
                       ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Plot 5: Confidence intervals
        original_ci = [results['original']['ci_lower'], results['original']['ci_upper']]
        cuped_ci = [results['cuped']['ci_lower'], results['cuped']['ci_upper']]

        axes[1, 1].errorbar([0], [results['original']['ate']],
                           yerr=[[results['original']['ate'] - original_ci[0]],
                                [original_ci[1] - results['original']['ate']]],
                           fmt='o', capsize=5, label='Original', color='orange', markersize=8)
        axes[1, 1].errorbar([1], [results['cuped']['ate']],
                           yerr=[[results['cuped']['ate'] - cuped_ci[0]],
                                [cuped_ci[1] - results['cuped']['ate']]],
                           fmt='s', capsize=5, label='CUPED', color='green', markersize=8)
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Treatment Effect')
        axes[1, 1].set_title('Confidence Intervals')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Plot 6: Summary statistics
        axes[1, 2].axis('off')
        summary = results['summary']

        summary_text = f"""
CUPED Summary Statistics

Variance Reduction: {summary['variance_reduction']:.1%}
Power Improvement: {summary['power_improvement']:.2f}×
SE Reduction: {summary['se_reduction']:.1%}

Adjustment R²: {summary['adjustment_r2']:.3f}
Sample Size: {summary['n_samples']:,}

Original SE: {results['original']['se']:.4f}
CUPED SE: {results['cuped']['se']:.4f}

Original p-value: {results['original']['p_value']:.4f}
CUPED p-value: {results['cuped']['p_value']:.4f}
        """

        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        return fig

    def generate_summary_report(self, outcome_col: str) -> str:
        """
        Generate a comprehensive summary report.
        
        Parameters:
        -----------
        outcome_col : str
            Name of outcome variable
            
        Returns:
        --------
        str
            Formatted summary report
        """
        if outcome_col not in self.results:
            raise ValueError(f"No results found for {outcome_col}. Run estimate_treatment_effects first.")

        results = self.results[outcome_col]
        adjustment_info = self.cuped_adjustments[outcome_col]

        report = "=" * 60 + "\n"
        report += "CUPED (Controlled-experiment Using Pre-Experiment Data) ANALYSIS\n"
        report += "=" * 60 + "\n\n"

        # Adjustment details
        report += "1. CUPED ADJUSTMENT DETAILS:\n"
        report += "-" * 30 + "\n"
        report += f"Adjustment method: {adjustment_info['method'].upper()}\n"
        report += f"Pre-experiment covariates: {', '.join(adjustment_info['covariate_cols'])}\n"
        report += f"Adjustment R²: {adjustment_info['r2']:.4f}\n"
        report += f"Expected variance reduction: {adjustment_info['variance_reduction']:.1%}\n"
        report += f"Estimation sample size: {adjustment_info['n_estimation']}\n\n"

        # Covariate balance
        if 'balance_check' in adjustment_info:
            report += "2. COVARIATE BALANCE CHECK:\n"
            report += "-" * 30 + "\n"
            balance = adjustment_info['balance_check']
            for covar, stats in balance.items():
                status = "✅ Balanced" if stats['balanced'] else "⚠️ Imbalanced"
                report += f"{covar}: {status} (std diff: {stats['std_diff']:.3f})\n"
            report += "\n"

        # Treatment effect results
        report += "3. TREATMENT EFFECT RESULTS:\n"
        report += "-" * 30 + "\n"

        orig = results['original']
        cuped = results['cuped']

        report += "Original Analysis:\n"
        report += f"  ATE: {orig['ate']:.4f}\n"
        report += f"  SE: {orig['se']:.4f}\n"
        report += f"  95% CI: [{orig['ci_lower']:.4f}, {orig['ci_upper']:.4f}]\n"
        report += f"  P-value: {orig['p_value']:.6f}\n\n"

        report += "CUPED-Adjusted Analysis:\n"
        report += f"  ATE: {cuped['ate']:.4f}\n"
        report += f"  SE: {cuped['se']:.4f}\n"
        report += f"  95% CI: [{cuped['ci_lower']:.4f}, {cuped['ci_upper']:.4f}]\n"
        report += f"  P-value: {cuped['p_value']:.6f}\n\n"

        # Improvement summary
        summary = results['summary']
        report += "4. CUPED IMPROVEMENTS:\n"
        report += "-" * 30 + "\n"
        report += f"Variance reduction: {summary['variance_reduction']:.1%}\n"
        report += f"Statistical power improvement: {summary['power_improvement']:.2f}×\n"
        report += f"Standard error reduction: {summary['se_reduction']:.1%}\n"
        report += f"Confidence interval width reduction: {summary['se_reduction']:.1%}\n\n"

        # Statistical significance
        original_sig = "significant" if orig['p_value'] < 0.05 else "not significant"
        cuped_sig = "significant" if cuped['p_value'] < 0.05 else "not significant"

        report += "5. STATISTICAL SIGNIFICANCE:\n"
        report += "-" * 30 + "\n"
        report += f"Original analysis: {original_sig} (p = {orig['p_value']:.6f})\n"
        report += f"CUPED analysis: {cuped_sig} (p = {cuped['p_value']:.6f})\n\n"

        if orig['p_value'] >= 0.05 and cuped['p_value'] < 0.05:
            report += "🎉 CUPED enabled detection of a significant effect!\n\n"
        elif orig['p_value'] < 0.05 and cuped['p_value'] >= 0.05:
            report += "⚠️ CUPED changed significance (investigate further)\n\n"
        elif cuped['p_value'] < orig['p_value']:
            report += "📈 CUPED improved statistical evidence\n\n"

        # Recommendations
        report += "6. RECOMMENDATIONS:\n"
        report += "-" * 30 + "\n"

        if summary['variance_reduction'] > 0.1:
            report += "✅ CUPED provides substantial variance reduction (>10%)\n"
            report += "✅ Recommended for future experiments with similar covariates\n"
        elif summary['variance_reduction'] > 0.05:
            report += "✅ CUPED provides moderate variance reduction (5-10%)\n"
            report += "✅ May be beneficial for larger experiments\n"
        else:
            report += "⚠️ Limited variance reduction (<5%)\n"
            report += "⚠️ Consider different or additional pre-experiment covariates\n"

        if adjustment_info['r2'] < 0.1:
            report += "⚠️ Low R² suggests weak covariate correlation\n"
            report += "💡 Consider including more relevant pre-experiment measures\n"

        report += "\n" + "=" * 60 + "\n"

        return report


def load_and_analyze_cuped(
    data_path: str,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list[str],
    adjustment_method: str = "ols"
) -> CUPED:
    """
    Load data and perform complete CUPED analysis.
    
    Parameters:
    -----------
    data_path : str
        Path to data file
    outcome_col : str
        Name of outcome variable
    treatment_col : str
        Name of treatment variable
    covariate_cols : List[str]
        Names of pre-experiment covariates
    adjustment_method : str
        Method for CUPED adjustment
        
    Returns:
    --------
    CUPED
        Fitted CUPED analyzer with results
    """
    # Load data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    # Initialize CUPED
    cuped = CUPED(data)

    # Run analysis
    cuped.estimate_treatment_effects(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        covariate_cols=covariate_cols,
        adjustment_method=adjustment_method
    )

    return cuped
