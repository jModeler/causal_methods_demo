"""
Causal Forest Implementation for Heterogeneous Treatment Effect Estimation

This module implements causal forests using the EconML library for estimating
heterogeneous treatment effects. Causal forests are particularly useful when
treatment effects vary across different subgroups of the population.

Key features:
- Heterogeneous treatment effect estimation
- Feature importance for treatment effect heterogeneity
- Confidence intervals for individual treatment effects
- Comprehensive visualization tools
- Model diagnostics and validation

Reference: Wager & Athey (2018), "Estimation and Inference of Heterogeneous
Treatment Effects using Random Forests"
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import econml for proper causal forest implementation
try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logger.warning("EconML not available. Using simplified causal forest implementation.")


class CausalForest:
    """
    Causal Forest implementation for heterogeneous treatment effect estimation.

    This class implements causal forests for estimating conditional average treatment
    effects (CATE) that can vary across different values of observed covariates.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing treatment, outcome, and covariates
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    n_estimators : int, optional
        Number of trees in the forest (default: 100)
    max_depth : int, optional
        Maximum depth of trees (default: None)
    min_samples_split : int, optional
        Minimum samples required to split a node (default: 5)
    min_samples_leaf : int, optional
        Minimum samples required at a leaf node (default: 5)
    """

    def __init__(self, data, random_state=42, n_estimators=100, max_depth=None,
                 min_samples_split=5, min_samples_leaf=5):
        self.data = data.copy()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Results storage
        self.treatment_effects = {}
        self.feature_importance = {}
        self.model_performance = {}

        np.random.seed(random_state)

    def fit_causal_forest(self, outcome_col, treatment_col, covariate_cols=None,
                         test_size=0.2, cv_folds=3):
        """
        Fit causal forest model to estimate heterogeneous treatment effects.

        Parameters
        ----------
        outcome_col : str
            Name of outcome variable column
        treatment_col : str
            Name of treatment variable column
        covariate_cols : list of str, optional
            List of covariate column names. If None, uses all numeric columns
        test_size : float, optional
            Proportion of data for testing (default: 0.2)
        cv_folds : int, optional
            Number of cross-validation folds (default: 3)

        Returns
        -------
        dict
            Dictionary containing model performance metrics and diagnostics
        """
        logger.info("Fitting causal forest model...")

        # Prepare data
        if covariate_cols is None:
            # Use all numeric columns except outcome and treatment
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            covariate_cols = [col for col in numeric_cols
                             if col not in [outcome_col, treatment_col, 'user_id']]

        # Extract variables
        y = self.data[outcome_col].values
        t = self.data[treatment_col].values
        X = self.data[covariate_cols].values

        # Handle binary outcomes
        self.is_binary_outcome = len(np.unique(y)) == 2
        if self.is_binary_outcome:
            y = y.astype(int)

        # Split data
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t, test_size=test_size, random_state=self.random_state,
            stratify=t if len(np.unique(t)) > 1 else None
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Store for later use
        self.covariate_cols = covariate_cols
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.t_test = t_test

        # Fit causal forest
        if ECONML_AVAILABLE:
            self._fit_econml_forest(X_train_scaled, y_train, t_train, X_test_scaled, y_test, t_test)
        else:
            self._fit_simple_forest(X_train_scaled, y_train, t_train, X_test_scaled, y_test, t_test)

        # Calculate feature importance for treatment effect heterogeneity
        self._calculate_feature_importance(X_train_scaled, y_train, t_train)

        # Estimate treatment effects
        self._estimate_treatment_effects(X_test_scaled, y_test, t_test)

        self.is_fitted = True
        logger.info("Causal forest model fitting completed.")

        return self.model_performance

    def _fit_econml_forest(self, X_train, y_train, t_train, X_test, y_test, t_test):
        """Fit causal forest using EconML library."""
        try:
            # Initialize CausalForestDML
            self.model = CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                ),
                model_t=RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                ),
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )

            # Fit the model
            self.model.fit(y_train, t_train, X=X_train)

            # Calculate performance metrics
            train_ate = self.model.ate(X_train)
            test_ate = self.model.ate(X_test)

            # Get treatment effect predictions
            train_te = self.model.effect(X_train)
            test_te = self.model.effect(X_test)

            # Store performance metrics
            self.model_performance = {
                'implementation': 'EconML CausalForestDML',
                'train_ate': float(train_ate),
                'test_ate': float(test_ate),
                'train_te_std': float(np.std(train_te)),
                'test_te_std': float(np.std(test_te)),
                'heterogeneity_measure': float(np.std(test_te) / (np.abs(np.mean(test_te)) + 1e-8))
            }

        except Exception as e:
            logger.warning(f"EconML implementation failed: {e}. Falling back to simple implementation.")
            self._fit_simple_forest(X_train, y_train, t_train, X_test, y_test, t_test)

    def _fit_simple_forest(self, X_train, y_train, t_train, X_test, y_test, t_test):
        """Fit simplified causal forest using separate models for treated/control."""
        logger.info("Using simplified causal forest implementation...")

        # Separate models for treated and control groups
        treated_mask = t_train == 1
        control_mask = t_train == 0

        if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
            raise ValueError("Need both treated and control units for causal forest")

        # Fit separate outcome models
        if self.is_binary_outcome:
            model_treated = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            model_control = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + 1
            )
        else:
            model_treated = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            model_control = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + 1
            )

        # Fit models
        model_treated.fit(X_train[treated_mask], y_train[treated_mask])
        model_control.fit(X_train[control_mask], y_train[control_mask])

        # Store models
        self.model = {
            'treated': model_treated,
            'control': model_control,
            'implementation': 'Simple T-learner'
        }

        # Calculate performance metrics
        if self.is_binary_outcome:
            treated_pred_train = model_treated.predict_proba(X_train)[:, 1]
            control_pred_train = model_control.predict_proba(X_train)[:, 1]
            treated_pred_test = model_treated.predict_proba(X_test)[:, 1]
            control_pred_test = model_control.predict_proba(X_test)[:, 1]
        else:
            treated_pred_train = model_treated.predict(X_train)
            control_pred_train = model_control.predict(X_train)
            treated_pred_test = model_treated.predict(X_test)
            control_pred_test = model_control.predict(X_test)

        # Calculate treatment effects
        train_te = treated_pred_train - control_pred_train
        test_te = treated_pred_test - control_pred_test

        self.model_performance = {
            'implementation': 'Simple T-learner',
            'train_ate': float(np.mean(train_te)),
            'test_ate': float(np.mean(test_te)),
            'train_te_std': float(np.std(train_te)),
            'test_te_std': float(np.std(test_te)),
            'heterogeneity_measure': float(np.std(test_te) / (np.abs(np.mean(test_te)) + 1e-8))
        }

    def _calculate_feature_importance(self, X_train, y_train, t_train):
        """Calculate feature importance for treatment effect heterogeneity."""
        if isinstance(self.model, dict):  # Simple implementation
            # Combine feature importance from both models
            treated_importance = self.model['treated'].feature_importances_
            control_importance = self.model['control'].feature_importances_
            combined_importance = (treated_importance + control_importance) / 2
        elif hasattr(self.model, 'feature_importances_'):
            combined_importance = self.model.feature_importances_
        else:
            # Fallback: calculate permutation importance
            combined_importance = self._calculate_permutation_importance(X_train, y_train, t_train)

        self.feature_importance = {
            'importances': combined_importance,
            'feature_names': self.covariate_cols,
            'sorted_indices': np.argsort(combined_importance)[::-1]
        }

    def _calculate_permutation_importance(self, X, y, t):
        """Calculate permutation-based feature importance."""
        baseline_score = self._calculate_baseline_score(X, y, t)
        importances = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self._calculate_baseline_score(X_permuted, y, t)
            importances[i] = baseline_score - permuted_score

        return np.maximum(importances, 0)  # Only positive importance

    def _calculate_baseline_score(self, X, y, t):
        """Calculate baseline score for permutation importance."""
        # Simple score based on treatment effect prediction accuracy
        if isinstance(self.model, dict):
            if self.is_binary_outcome:
                treated_pred = self.model['treated'].predict_proba(X)[:, 1]
                control_pred = self.model['control'].predict_proba(X)[:, 1]
            else:
                treated_pred = self.model['treated'].predict(X)
                control_pred = self.model['control'].predict(X)

            predicted_te = treated_pred - control_pred

            # Calculate "true" treatment effect as difference in observed outcomes
            treated_mask = t == 1
            control_mask = t == 0

            if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
                observed_te = np.mean(y[treated_mask]) - np.mean(y[control_mask])
                score = -np.mean((predicted_te - observed_te) ** 2)
            else:
                score = -np.var(predicted_te)  # Prefer less variable predictions
        else:
            score = 0  # Fallback for EconML

        return score

    def _estimate_treatment_effects(self, X_test, y_test, t_test):
        """Estimate individual and average treatment effects."""
        if isinstance(self.model, dict):  # Simple implementation
            if self.is_binary_outcome:
                treated_pred = self.model['treated'].predict_proba(X_test)[:, 1]
                control_pred = self.model['control'].predict_proba(X_test)[:, 1]
            else:
                treated_pred = self.model['treated'].predict(X_test)
                control_pred = self.model['control'].predict(X_test)

            individual_te = treated_pred - control_pred

            # Calculate confidence intervals using bootstrap
            cis = self._bootstrap_confidence_intervals(X_test, n_bootstrap=100)

        else:  # EconML implementation
            individual_te = self.model.effect(X_test)
            # Try to get confidence intervals
            try:
                cis = self.model.effect_interval(X_test, alpha=0.05)
                cis = {'lower': cis[0], 'upper': cis[1]}
            except (AttributeError, ValueError):
                cis = self._bootstrap_confidence_intervals(X_test, n_bootstrap=100)

        # Calculate statistics
        ate = np.mean(individual_te)
        ate_se = np.std(individual_te) / np.sqrt(len(individual_te))

        # Store results
        self.treatment_effects = {
            'individual_effects': individual_te,
            'ate': ate,
            'ate_se': ate_se,
            'ate_ci_lower': ate - 1.96 * ate_se,
            'ate_ci_upper': ate + 1.96 * ate_se,
            'individual_ci_lower': cis.get('lower', individual_te - 1.96 * ate_se),
            'individual_ci_upper': cis.get('upper', individual_te + 1.96 * ate_se),
            'heterogeneity_std': np.std(individual_te),
            'p_value': 2 * (1 - stats.norm.cdf(np.abs(ate / (ate_se + 1e-8))))
        }

    def _bootstrap_confidence_intervals(self, X_test, n_bootstrap=100, alpha=0.05):
        """Calculate bootstrap confidence intervals for treatment effects."""
        bootstrap_tes = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            n_samples = X_test.shape[0]
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_test[bootstrap_indices]

            # Predict treatment effects
            if isinstance(self.model, dict):
                if self.is_binary_outcome:
                    treated_pred = self.model['treated'].predict_proba(X_bootstrap)[:, 1]
                    control_pred = self.model['control'].predict_proba(X_bootstrap)[:, 1]
                else:
                    treated_pred = self.model['treated'].predict(X_bootstrap)
                    control_pred = self.model['control'].predict(X_bootstrap)
                te_bootstrap = treated_pred - control_pred
            else:
                te_bootstrap = self.model.effect(X_bootstrap)

            bootstrap_tes.append(te_bootstrap)

        bootstrap_tes = np.array(bootstrap_tes)

        # Calculate percentile confidence intervals
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_tes, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_tes, upper_percentile, axis=0)

        return {'lower': ci_lower, 'upper': ci_upper}

    def estimate_conditional_effects(self, feature_values_dict):
        """
        Estimate treatment effects for specific feature values.

        Parameters
        ----------
        feature_values_dict : dict
            Dictionary mapping feature names to values

        Returns
        -------
        dict
            Treatment effect estimates for specified conditions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before estimating conditional effects")

        # Create feature vector
        X_cond = np.zeros((1, len(self.covariate_cols)))
        for i, feature in enumerate(self.covariate_cols):
            if feature in feature_values_dict:
                X_cond[0, i] = feature_values_dict[feature]
            else:
                # Use median value for unspecified features
                median_val = np.median(self.scaler.inverse_transform(self.X_test)[:, i])
                X_cond[0, i] = median_val

        # Scale features
        X_cond_scaled = self.scaler.transform(X_cond)

        # Predict treatment effect
        if isinstance(self.model, dict):
            if self.is_binary_outcome:
                treated_pred = self.model['treated'].predict_proba(X_cond_scaled)[:, 1]
                control_pred = self.model['control'].predict_proba(X_cond_scaled)[:, 1]
            else:
                treated_pred = self.model['treated'].predict(X_cond_scaled)
                control_pred = self.model['control'].predict(X_cond_scaled)
            conditional_te = treated_pred[0] - control_pred[0]
        else:
            conditional_te = self.model.effect(X_cond_scaled)[0]

        return {
            'conditional_treatment_effect': conditional_te,
            'feature_values': feature_values_dict,
            'unspecified_features_used_median': [f for f in self.covariate_cols
                                               if f not in feature_values_dict]
        }

    def plot_treatment_effect_distribution(self, bins=30, figsize=(12, 8)):
        """
        Create comprehensive visualizations of treatment effect heterogeneity.

        Parameters
        ----------
        bins : int, optional
            Number of bins for histogram (default: 30)
        figsize : tuple, optional
            Figure size (default: (12, 8))
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Causal Forest: Treatment Effect Heterogeneity Analysis', fontsize=16)

        individual_te = self.treatment_effects['individual_effects']
        ate = self.treatment_effects['ate']

        # Plot 1: Distribution of individual treatment effects
        axes[0, 0].hist(individual_te, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(ate, color='red', linestyle='--', linewidth=2,
                          label=f'ATE: {ate:.4f}')
        axes[0, 0].set_xlabel('Individual Treatment Effect')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Individual Treatment Effects')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Treatment effects with confidence intervals
        sorted_indices = np.argsort(individual_te)
        sorted_te = individual_te[sorted_indices]
        sorted_ci_lower = self.treatment_effects['individual_ci_lower'][sorted_indices]
        sorted_ci_upper = self.treatment_effects['individual_ci_upper'][sorted_indices]

        x_range = np.arange(len(sorted_te))
        axes[0, 1].fill_between(x_range, sorted_ci_lower, sorted_ci_upper,
                               alpha=0.3, color='lightblue', label='95% CI')
        axes[0, 1].plot(x_range, sorted_te, color='navy', linewidth=1, label='Individual TE')
        axes[0, 1].axhline(ate, color='red', linestyle='--', linewidth=2, label=f'ATE: {ate:.4f}')
        axes[0, 1].set_xlabel('Individual (sorted by effect size)')
        axes[0, 1].set_ylabel('Treatment Effect')
        axes[0, 1].set_title('Individual Treatment Effects with Confidence Intervals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Feature importance for treatment effect heterogeneity
        importance_data = self.feature_importance
        sorted_idx = importance_data['sorted_indices'][:10]  # Top 10 features
        top_features = [importance_data['feature_names'][i] for i in sorted_idx]
        top_importance = importance_data['importances'][sorted_idx]

        bars = axes[1, 0].barh(range(len(top_features)), top_importance, color='lightcoral')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top Features for Treatment Effect Heterogeneity')
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for _, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 0].text(width, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center', fontsize=9)

        # Plot 4: Treatment effect vs most important feature
        if len(self.covariate_cols) > 0:
            most_important_feature_idx = importance_data['sorted_indices'][0]
            feature_values = self.scaler.inverse_transform(self.X_test)[:, most_important_feature_idx]

            axes[1, 1].scatter(feature_values, individual_te, alpha=0.6, color='green')

            # Add trend line
            z = np.polyfit(feature_values, individual_te, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(sorted(feature_values), p(sorted(feature_values)),
                           "r--", alpha=0.8, linewidth=2)

            axes[1, 1].set_xlabel(f'{importance_data["feature_names"][most_important_feature_idx]}')
            axes[1, 1].set_ylabel('Treatment Effect')
            axes[1, 1].set_title(f'Treatment Effect vs {importance_data["feature_names"][most_important_feature_idx]}')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=10, figsize=(10, 6)):
        """
        Plot feature importance for treatment effect heterogeneity.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to show (default: 10)
        figsize : tuple, optional
            Figure size (default: (10, 6))
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")

        importance_data = self.feature_importance
        sorted_idx = importance_data['sorted_indices'][:top_n]
        top_features = [importance_data['feature_names'][i] for i in sorted_idx]
        top_importance = importance_data['importances'][sorted_idx]

        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_importance, color='lightcoral')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features for Treatment Effect Heterogeneity')
        plt.grid(True, alpha=0.3)

        # Add value labels
        for _, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.show()

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of causal forest analysis.

        Returns
        -------
        str
            Formatted summary report
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating report")

        ate = self.treatment_effects['ate']
        ate_se = self.treatment_effects['ate_se']
        p_value = self.treatment_effects['p_value']
        heterogeneity_std = self.treatment_effects['heterogeneity_std']

        report = "=" * 60 + "\n"
        report += "CAUSAL FOREST ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"

        # Model Information
        report += " MODEL INFORMATION\n"
        report += "-" * 30 + "\n"
        report += f"Implementation: {self.model_performance['implementation']}\n"
        report += f"Number of trees: {self.n_estimators}\n"
        report += f"Outcome variable: {self.outcome_col}\n"
        report += f"Treatment variable: {self.treatment_col}\n"
        report += f"Number of covariates: {len(self.covariate_cols)}\n"
        report += f"Sample size: {len(self.data)}\n\n"

        # Treatment Effect Results
        report += " TREATMENT EFFECT RESULTS\n"
        report += "-" * 30 + "\n"
        report += f"Average Treatment Effect (ATE): {ate:.4f}\n"
        report += f"Standard Error: {ate_se:.4f}\n"
        report += f"95% Confidence Interval: [{self.treatment_effects['ate_ci_lower']:.4f}, "
        report += f"{self.treatment_effects['ate_ci_upper']:.4f}]\n"
        report += f"P-value: {p_value:.4f}\n"
        significance = "Yes" if p_value < 0.05 else "No"
        report += f"Statistically Significant (α=0.05): {significance}\n\n"

        # Heterogeneity Analysis
        report += " HETEROGENEITY ANALYSIS\n"
        report += "-" * 30 + "\n"
        report += f"Treatment Effect Standard Deviation: {heterogeneity_std:.4f}\n"
        report += f"Heterogeneity Ratio: {self.model_performance['heterogeneity_measure']:.4f}\n"

        # Interpret heterogeneity
        if self.model_performance['heterogeneity_measure'] > 0.5:
            report += "Interpretation: HIGH heterogeneity - treatment effects vary substantially\n"
        elif self.model_performance['heterogeneity_measure'] > 0.2:
            report += "Interpretation: MODERATE heterogeneity - some variation in treatment effects\n"
        else:
            report += "Interpretation: LOW heterogeneity - treatment effects are relatively homogeneous\n"

        # Effect distribution
        individual_effects = self.treatment_effects['individual_effects']
        positive_effects = np.sum(individual_effects > 0)
        negative_effects = np.sum(individual_effects < 0)
        total_effects = len(individual_effects)

        report += "\nEffect Distribution:\n"
        report += f"- Positive effects: {positive_effects}/{total_effects} ({100*positive_effects/total_effects:.1f}%)\n"
        report += f"- Negative effects: {negative_effects}/{total_effects} ({100*negative_effects/total_effects:.1f}%)\n"
        report += f"- Range: [{np.min(individual_effects):.4f}, {np.max(individual_effects):.4f}]\n\n"

        # Top Features
        report += " TOP FEATURES FOR HETEROGENEITY\n"
        report += "-" * 30 + "\n"
        importance_data = self.feature_importance
        sorted_idx = importance_data['sorted_indices'][:5]  # Top 5

        for i, idx in enumerate(sorted_idx, 1):
            feature_name = importance_data['feature_names'][idx]
            importance = importance_data['importances'][idx]
            report += f"{i}. {feature_name}: {importance:.4f}\n"

        # Business Recommendations
        report += "\n BUSINESS RECOMMENDATIONS\n"
        report += "-" * 30 + "\n"

        if p_value < 0.05:
            if ate > 0:
                report += " Treatment has a positive and statistically significant effect\n"
                if self.model_performance['heterogeneity_measure'] > 0.3:
                    report += " High heterogeneity suggests targeting specific subgroups for maximum impact\n"
                    top_feature = importance_data['feature_names'][importance_data['sorted_indices'][0]]
                    report += f" Focus on segmentation by: {top_feature}\n"
                else:
                    report += " Consider broad implementation due to consistent positive effects\n"
            else:
                report += "️  Treatment has a negative effect - reconsider implementation\n"
        else:
            report += " Treatment effect is not statistically significant\n"
            report += " Consider collecting more data or refining the intervention\n"

        # Model Performance
        report += "\n MODEL PERFORMANCE\n"
        report += "-" * 30 + "\n"
        report += f"Train ATE: {self.model_performance['train_ate']:.4f}\n"
        report += f"Test ATE: {self.model_performance['test_ate']:.4f}\n"
        report += f"ATE Stability: {'Good' if abs(self.model_performance['train_ate'] - self.model_performance['test_ate']) < 0.1 * abs(ate) else 'Poor'}\n"

        return report
