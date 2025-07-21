"""
Difference-in-Differences (DiD) implementation for causal inference.

This module implements DiD analysis for estimating the causal effect of the 
Smart Filing Assistant on user conversion and engagement metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for causal inference.
    
    This class implements DiD methodology for panel data with pre and post
    treatment periods. Suitable for analyzing the Smart Filing Assistant
    impact on tax software users.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DiD estimator.
        
        Args:
            data: DataFrame containing user data with treatment and outcomes
        """
        self.data = data.copy()
        self.results = {}
        self.fitted_models = {}
        
    def prepare_panel_data(self, 
                          user_id_col: str = 'user_id',
                          treatment_col: str = 'used_smart_assistant',
                          outcome_2023_col: str = 'filed_2023', 
                          outcome_2024_col: str = 'filed_2024') -> pd.DataFrame:
        """
        Reshape data from wide to long format for DiD analysis.
        
        Args:
            user_id_col: Column name for user identifier
            treatment_col: Column name for treatment indicator
            outcome_2023_col: Column name for 2023 outcome
            outcome_2024_col: Column name for 2024 outcome
            
        Returns:
            Long format DataFrame suitable for DiD regression
        """
        # Select relevant columns
        cols_to_keep = [user_id_col, treatment_col, outcome_2023_col, outcome_2024_col]
        
        # Add control variables if they exist
        control_vars = ['age', 'income_bracket', 'device_type', 'user_type', 'region', 'tech_savviness']
        for var in control_vars:
            if var in self.data.columns:
                cols_to_keep.append(var)
        
        subset_data = self.data[cols_to_keep].copy()
        
        # Reshape to long format
        long_data = []
        
        for _, row in subset_data.iterrows():
            # 2023 observation (pre-treatment)
            obs_2023 = {
                user_id_col: row[user_id_col],
                'year': 2023,
                'post_treatment': 0,
                'treated': row[treatment_col],
                'outcome': row[outcome_2023_col],
                'treatment_x_post': 0  # No treatment effect in pre-period
            }
            
            # Add control variables
            for var in control_vars:
                if var in row.index:
                    obs_2023[var] = row[var]
            
            long_data.append(obs_2023)
            
            # 2024 observation (post-treatment)
            obs_2024 = {
                user_id_col: row[user_id_col],
                'year': 2024,
                'post_treatment': 1,
                'treated': row[treatment_col],
                'outcome': row[outcome_2024_col],
                'treatment_x_post': row[treatment_col] * 1  # Treatment effect
            }
            
            # Add control variables
            for var in control_vars:
                if var in row.index:
                    obs_2024[var] = row[var]
            
            long_data.append(obs_2024)
        
        panel_df = pd.DataFrame(long_data)
        self.panel_data = panel_df
        return panel_df
    
    def estimate_did(self, 
                    outcome_col: str = 'outcome',
                    control_vars: Optional[List[str]] = None,
                    cluster_se: bool = True) -> Dict[str, Any]:
        """
        Estimate DiD treatment effect using regression.
        
        Args:
            outcome_col: Name of outcome variable
            control_vars: List of control variables to include
            cluster_se: Whether to cluster standard errors by user
            
        Returns:
            Dictionary containing estimation results
        """
        if not hasattr(self, 'panel_data'):
            raise ValueError("Must call prepare_panel_data() first")
        
        # Build regression formula
        formula = f"{outcome_col} ~ treated + post_treatment + treatment_x_post"
        
        if control_vars:
            # Add control variables
            available_controls = [var for var in control_vars if var in self.panel_data.columns]
            if available_controls:
                formula += " + " + " + ".join(available_controls)
        
        # Fit the model
        if cluster_se:
            # Use robust standard errors clustered by user
            model = smf.ols(formula, data=self.panel_data).fit(
                cov_type='cluster', cov_kwds={'groups': self.panel_data['user_id']}
            )
        else:
            model = smf.ols(formula, data=self.panel_data).fit()
        
        # Extract DiD coefficient (treatment_x_post)
        did_coef = model.params['treatment_x_post']
        did_se = model.bse['treatment_x_post']
        did_pvalue = model.pvalues['treatment_x_post']
        
        # Confidence interval
        conf_int = model.conf_int().loc['treatment_x_post']
        
        results = {
            'did_estimate': did_coef,
            'standard_error': did_se,
            'p_value': did_pvalue,
            'conf_int_lower': conf_int[0],
            'conf_int_upper': conf_int[1],
            'model': model,
            'formula': formula,
            'n_observations': len(self.panel_data),
            'n_users': self.panel_data['user_id'].nunique()
        }
        
        self.results['main'] = results
        self.fitted_models['main'] = model
        return results
    
    def parallel_trends_test(self, 
                           pre_periods: Optional[List[int]] = None,
                           outcome_col: str = 'outcome') -> Dict[str, Any]:
        """
        Test the parallel trends assumption (if we had multiple pre-periods).
        
        For this demo with only 2023-2024 data, we'll create a placebo test
        by splitting 2023 data and checking if treatment predicts the change.
        
        Args:
            pre_periods: List of years to test (not applicable in our 2-period case)
            outcome_col: Name of outcome variable
            
        Returns:
            Dictionary with parallel trends test results
        """
        print("Note: With only 2 time periods (2023, 2024), formal parallel trends testing is limited.")
        print("In practice, you would need multiple pre-treatment periods.")
        
        # Alternative: Check if treatment is correlated with 2023 baseline levels
        baseline_data = self.data[['used_smart_assistant', 'filed_2023', 'time_to_complete_2023', 
                                  'sessions_2023', 'support_tickets_2023']].copy()
        
        correlation_results = {}
        
        for outcome in ['filed_2023', 'time_to_complete_2023', 'sessions_2023', 'support_tickets_2023']:
            if outcome in baseline_data.columns:
                # Only include users who filed in 2023 for time/sessions variables
                if outcome in ['time_to_complete_2023', 'sessions_2023']:
                    subset = baseline_data[baseline_data['filed_2023'] == 1]
                else:
                    subset = baseline_data
                
                corr, p_val = stats.pearsonr(subset['used_smart_assistant'], subset[outcome])
                correlation_results[outcome] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
        
        return {
            'test_type': 'baseline_correlation',
            'description': 'Correlation between treatment and pre-treatment outcomes',
            'results': correlation_results,
            'interpretation': 'High correlations suggest potential parallel trends violations'
        }
    
    def estimate_heterogeneous_effects(self, 
                                     subgroup_var: str,
                                     outcome_col: str = 'outcome') -> Dict[str, Any]:
        """
        Estimate heterogeneous treatment effects by subgroups.
        
        Args:
            subgroup_var: Variable to define subgroups (e.g., 'age_group', 'income_bracket')
            outcome_col: Name of outcome variable
            
        Returns:
            Dictionary containing subgroup-specific estimates
        """
        if not hasattr(self, 'panel_data'):
            raise ValueError("Must call prepare_panel_data() first")
        
        # Add subgroup variable to panel data if not present
        if subgroup_var not in self.panel_data.columns:
            # Create it from original data
            if subgroup_var == 'age_group':
                self.data['age_group'] = pd.cut(self.data['age'], 
                                              bins=[0, 35, 50, 65, 100], 
                                              labels=['<35', '35-50', '50-65', '65+'])
                # Merge back to panel data
                age_mapping = self.data[['user_id', 'age_group']].drop_duplicates()
                self.panel_data = self.panel_data.merge(age_mapping, on='user_id', how='left')
            else:
                # For other variables, merge from original data
                var_mapping = self.data[['user_id', subgroup_var]].drop_duplicates()
                self.panel_data = self.panel_data.merge(var_mapping, on='user_id', how='left')
        
        subgroup_results = {}
        
        # Get unique subgroup values
        subgroups = self.panel_data[subgroup_var].unique()
        
        for subgroup in subgroups:
            if pd.isna(subgroup):
                continue
                
            # Filter data for this subgroup
            subgroup_data = self.panel_data[self.panel_data[subgroup_var] == subgroup]
            
            if len(subgroup_data) < 20:  # Skip if too few observations
                continue
            
            # Estimate DiD for this subgroup
            formula = f"{outcome_col} ~ treated + post_treatment + treatment_x_post"
            
            try:
                model = smf.ols(formula, data=subgroup_data).fit()
                
                did_coef = model.params['treatment_x_post']
                did_se = model.bse['treatment_x_post']
                did_pvalue = model.pvalues['treatment_x_post']
                conf_int = model.conf_int().loc['treatment_x_post']
                
                subgroup_results[str(subgroup)] = {
                    'did_estimate': did_coef,
                    'standard_error': did_se,
                    'p_value': did_pvalue,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'n_observations': len(subgroup_data),
                    'n_users': subgroup_data['user_id'].nunique()
                }
                
            except Exception as e:
                print(f"Could not estimate model for subgroup {subgroup}: {e}")
                continue
        
        return {
            'subgroup_variable': subgroup_var,
            'results': subgroup_results
        }
    
    def plot_parallel_trends(self, 
                           outcome_2023_col: str = 'filed_2023',
                           outcome_2024_col: str = 'filed_2024',
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot parallel trends visualization.
        
        Args:
            outcome_2023_col: Column name for 2023 outcome
            outcome_2024_col: Column name for 2024 outcome
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        # Calculate means by treatment group and year
        treated_2023 = self.data[self.data['used_smart_assistant'] == 1][outcome_2023_col].mean()
        treated_2024 = self.data[self.data['used_smart_assistant'] == 1][outcome_2024_col].mean()
        control_2023 = self.data[self.data['used_smart_assistant'] == 0][outcome_2023_col].mean()
        control_2024 = self.data[self.data['used_smart_assistant'] == 0][outcome_2024_col].mean()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        years = [2023, 2024]
        treated_means = [treated_2023, treated_2024]
        control_means = [control_2023, control_2024]
        
        # Plot lines
        ax.plot(years, treated_means, 'o-', linewidth=2, markersize=8, 
                label='Treated (Used Smart Assistant)', color='blue')
        ax.plot(years, control_means, 's-', linewidth=2, markersize=8, 
                label='Control (No Smart Assistant)', color='red')
        
        # Add vertical line at treatment start
        ax.axvline(x=2023.5, color='gray', linestyle='--', alpha=0.7, 
                   label='Smart Assistant Launch')
        
        # Formatting
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{outcome_2023_col.replace("_", " ").title()} Rate')
        ax.set_title('Parallel Trends: Treatment vs Control Groups')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(years)
        
        # Add text box with DiD estimate if available
        if 'main' in self.results:
            did_est = self.results['main']['did_estimate']
            ax.text(0.02, 0.98, f'DiD Estimate: {did_est:.3f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_subgroup_effects(self, 
                            subgroup_results: Dict[str, Any],
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot heterogeneous treatment effects by subgroups.
        
        Args:
            subgroup_results: Results from estimate_heterogeneous_effects()
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        results = subgroup_results['results']
        
        if not results:
            print("No subgroup results to plot")
            return None
        
        # Extract data for plotting
        subgroups = list(results.keys())
        estimates = [results[sg]['did_estimate'] for sg in subgroups]
        lower_ci = [results[sg]['conf_int_lower'] for sg in subgroups]
        upper_ci = [results[sg]['conf_int_upper'] for sg in subgroups]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(subgroups))
        
        # Plot point estimates with confidence intervals
        ax.errorbar(estimates, y_pos, 
                   xerr=[np.array(estimates) - np.array(lower_ci),
                         np.array(upper_ci) - np.array(estimates)],
                   fmt='o', markersize=8, capsize=5, capthick=2)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subgroups)
        ax.set_xlabel('DiD Treatment Effect')
        ax.set_title(f'Heterogeneous Treatment Effects by {subgroup_results["subgroup_variable"]}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (est, sg) in enumerate(zip(estimates, subgroups)):
            ax.text(est + 0.01, i, f'{est:.3f}', 
                   verticalalignment='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def summary_report(self) -> str:
        """
        Generate a summary report of DiD results.
        
        Returns:
            Formatted string with key findings
        """
        if 'main' not in self.results:
            return "No DiD results available. Run estimate_did() first."
        
        main_results = self.results['main']
        
        report = "="*60 + "\n"
        report += "DIFFERENCE-IN-DIFFERENCES ANALYSIS SUMMARY\n"
        report += "="*60 + "\n\n"
        
        report += f"Treatment Effect Estimate: {main_results['did_estimate']:.4f}\n"
        report += f"Standard Error: {main_results['standard_error']:.4f}\n"
        report += f"P-value: {main_results['p_value']:.4f}\n"
        report += f"95% Confidence Interval: [{main_results['conf_int_lower']:.4f}, {main_results['conf_int_upper']:.4f}]\n\n"
        
        # Statistical significance
        if main_results['p_value'] < 0.001:
            significance = "highly significant (p < 0.001)"
        elif main_results['p_value'] < 0.01:
            significance = "significant (p < 0.01)"
        elif main_results['p_value'] < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not statistically significant (p >= 0.05)"
        
        report += f"Statistical Significance: {significance}\n\n"
        
        # Sample size
        report += f"Sample Size: {main_results['n_users']:,} users, {main_results['n_observations']:,} observations\n\n"
        
        # Interpretation
        report += "INTERPRETATION:\n"
        report += "-" * 40 + "\n"
        
        effect_pct = main_results['did_estimate'] * 100
        if main_results['did_estimate'] > 0:
            direction = "increased"
        else:
            direction = "decreased"
            effect_pct = abs(effect_pct)
        
        report += f"The Smart Filing Assistant {direction} the filing rate by "
        report += f"{effect_pct:.1f} percentage points on average.\n\n"
        
        if main_results['p_value'] < 0.05:
            report += "This effect is statistically significant, suggesting the Smart Filing Assistant "
            report += "has a genuine causal impact on user conversion rates.\n"
        else:
            report += "This effect is not statistically significant, so we cannot confidently "
            report += "conclude that the Smart Filing Assistant has a causal impact.\n"
        
        return report


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare data for DiD analysis.
    
    Args:
        file_path: Path to the CSV data file
        
    Returns:
        Prepared DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Convert boolean columns if needed
    bool_cols = ['filed_2023', 'filed_2024', 'used_smart_assistant', 'early_login_2024']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df 