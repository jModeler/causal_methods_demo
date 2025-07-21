"""
Propensity Score Matching (PSM) Implementation.

This module implements propensity score matching for causal inference, specifically
designed for analyzing the impact of the Smart Filing Assistant on user conversion
and engagement metrics.

Key features:
- Propensity score estimation using logistic regression
- Multiple matching algorithms (nearest neighbor, caliper, optimal)
- Balance assessment and diagnostics
- Treatment effect estimation
- Comprehensive visualization tools
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class PropensityScoreMatching:
    """
    Propensity Score Matching for causal inference.

    This class implements propensity score matching to estimate treatment effects
    by matching treated and control units based on their propensity scores.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize PSM with data.

        Args:
            data: DataFrame with treatment, outcome, and covariate columns
        """
        self.data = data.copy()
        self.propensity_scores = None
        self.matched_data = None
        self.balance_stats = None
        self.treatment_effects = None
        self.model = None
        self.scaler = StandardScaler()

    def estimate_propensity_scores(
        self,
        treatment_col: str = "used_smart_assistant",
        covariates: list[str] | None = None,
        include_interactions: bool = False,
        model_params: dict | None = None,
    ) -> dict[str, Any]:
        """
        Estimate propensity scores using logistic regression.

        Args:
            treatment_col: Name of treatment variable
            covariates: List of covariate column names
            include_interactions: Whether to include interaction terms
            model_params: Parameters for logistic regression

        Returns:
            Dictionary with model results and diagnostics
        """
        if covariates is None:
            # Default covariates for tax software data
            covariates = [
                "age",
                "tech_savviness",
                "income_bracket",
                "device_type",
                "user_type",
                "region",
                "filed_2023",
                "early_login_2024",
            ]

        # Filter covariates that exist in data
        available_covariates = [col for col in covariates if col in self.data.columns]

        if not available_covariates:
            raise ValueError("No valid covariates found in data")

        # Prepare features
        X = self._prepare_features(available_covariates, include_interactions)
        y = self.data[treatment_col]

        # Fit propensity score model
        model_params = model_params or {"random_state": 42, "max_iter": 1000}
        self.model = LogisticRegression(**model_params)
        self.model.fit(X, y)

        # Calculate propensity scores
        propensity_scores = self.model.predict_proba(X)[:, 1]
        self.propensity_scores = propensity_scores

        # Add to data
        self.data["propensity_score"] = propensity_scores
        self.data["predicted_treatment"] = self.model.predict(X)

        # Model diagnostics
        y_pred = self.model.predict(X)
        y_pred_proba = propensity_scores

        # Get classification report with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Precision is ill-defined", category=UserWarning
            )
            classification_dict = classification_report(
                y, y_pred, output_dict=True, zero_division=0
            )

        results = {
            "model": self.model,
            "covariates_used": available_covariates,
            "auc_score": roc_auc_score(y, y_pred_proba),
            "classification_report": classification_dict,
            "propensity_score_range": {
                "min": propensity_scores.min(),
                "max": propensity_scores.max(),
                "mean": propensity_scores.mean(),
                "std": propensity_scores.std(),
            },
            "common_support": self._assess_common_support(y, propensity_scores),
            "feature_importance": dict(
                zip(
                    self._get_feature_names(available_covariates, include_interactions),
                    self.model.coef_[0],
                    strict=False,
                )
            ),
        }

        return results

    def perform_matching(
        self,
        method: str = "nearest_neighbor",
        caliper: float | None = None,
        replacement: bool = False,
        ratio: int = 1,
        treatment_col: str = "used_smart_assistant",
    ) -> dict[str, Any]:
        """
        Perform propensity score matching.

        Args:
            method: Matching method ('nearest_neighbor', 'caliper', 'optimal')
            caliper: Maximum distance for matching (e.g., 0.1 * std of propensity scores)
            replacement: Whether to allow replacement in matching
            ratio: Number of control units to match to each treated unit
            treatment_col: Name of treatment variable

        Returns:
            Dictionary with matching results and diagnostics
        """
        if self.propensity_scores is None:
            raise ValueError("Must estimate propensity scores first")

        treated_mask = self.data[treatment_col] == 1
        control_mask = self.data[treatment_col] == 0

        treated_indices = self.data[treated_mask].index
        control_indices = self.data[control_mask].index

        treated_ps = self.propensity_scores[treated_mask]
        control_ps = self.propensity_scores[control_mask]

        if method == "nearest_neighbor":
            matches = self._nearest_neighbor_matching(
                treated_indices,
                control_indices,
                treated_ps,
                control_ps,
                caliper,
                replacement,
                ratio,
            )
        elif method == "caliper":
            if caliper is None:
                caliper = 0.1 * np.std(self.propensity_scores)
            matches = self._caliper_matching(
                treated_indices, control_indices, treated_ps, control_ps, caliper
            )
        else:
            raise ValueError(f"Unknown matching method: {method}")

        # Create matched dataset
        matched_treated_idx = []
        matched_control_idx = []

        for treated_idx, control_match_list in matches.items():
            if control_match_list:  # If matches found
                matched_treated_idx.extend([treated_idx] * len(control_match_list))
                matched_control_idx.extend(control_match_list)

        # Combine matched units
        matched_indices = matched_treated_idx + matched_control_idx
        self.matched_data = self.data.loc[matched_indices].copy()
        self.matched_data["match_group"] = list(range(len(matched_treated_idx))) + list(
            range(len(matched_control_idx))
        )

        # Calculate matching statistics
        n_treated_matched = len(set(matched_treated_idx))
        n_control_matched = len(matched_control_idx)
        n_treated_total = treated_mask.sum()

        results = {
            "method": method,
            "matches": matches,
            "n_treated_total": n_treated_total,
            "n_treated_matched": n_treated_matched,
            "n_control_matched": n_control_matched,
            "matching_rate": n_treated_matched / n_treated_total,
            "matched_data_shape": self.matched_data.shape,
            "average_matches_per_treated": n_control_matched
            / max(n_treated_matched, 1),
            "caliper_used": caliper,
        }

        return results

    def assess_balance(
        self,
        covariates: list[str] | None = None,
        treatment_col: str = "used_smart_assistant",
    ) -> dict[str, Any]:
        """
        Assess covariate balance before and after matching.

        Args:
            covariates: List of covariates to assess
            treatment_col: Name of treatment variable

        Returns:
            Dictionary with balance statistics
        """
        if covariates is None:
            covariates = [
                "age",
                "tech_savviness",
                "filed_2023",
                "early_login_2024",
                "propensity_score",
            ]

        # Filter available covariates
        available_covariates = [col for col in covariates if col in self.data.columns]

        # Calculate balance before matching
        before_balance = self._calculate_balance_stats(
            self.data, available_covariates, treatment_col
        )

        # Calculate balance after matching (if matched data exists)
        after_balance = None
        if self.matched_data is not None:
            after_balance = self._calculate_balance_stats(
                self.matched_data, available_covariates, treatment_col
            )

        self.balance_stats = {
            "before_matching": before_balance,
            "after_matching": after_balance,
            "covariates_assessed": available_covariates,
        }

        return self.balance_stats

    def estimate_treatment_effects(
        self,
        outcome_cols: str | list[str] = "filed_2024",
        treatment_col: str = "used_smart_assistant",
        method: str = "simple_difference",
    ) -> dict[str, Any]:
        """
        Estimate treatment effects on matched sample.

        Args:
            outcome_cols: Outcome variable(s) to analyze
            treatment_col: Name of treatment variable
            method: Method for effect estimation

        Returns:
            Dictionary with treatment effect estimates
        """
        if self.matched_data is None:
            raise ValueError("Must perform matching first")

        if isinstance(outcome_cols, str):
            outcome_cols = [outcome_cols]

        # Filter available outcomes
        available_outcomes = [
            col for col in outcome_cols if col in self.matched_data.columns
        ]

        if not available_outcomes:
            raise ValueError("No valid outcome variables found")

        results = {}

        for outcome in available_outcomes:
            if method == "simple_difference":
                effect_stats = self._simple_difference_estimator(
                    self.matched_data, outcome, treatment_col
                )
            else:
                raise ValueError(f"Unknown estimation method: {method}")

            results[outcome] = effect_stats

        self.treatment_effects = results
        return results

    def plot_propensity_distribution(
        self,
        treatment_col: str = "used_smart_assistant",
        figsize: tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """Plot propensity score distributions by treatment group."""
        if self.propensity_scores is None:
            raise ValueError("Must estimate propensity scores first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Distribution plot
        treated = self.data[self.data[treatment_col] == 1]["propensity_score"]
        control = self.data[self.data[treatment_col] == 0]["propensity_score"]

        ax1.hist(control, bins=30, alpha=0.7, label="Control", density=True)
        ax1.hist(treated, bins=30, alpha=0.7, label="Treated", density=True)
        ax1.set_xlabel("Propensity Score")
        ax1.set_ylabel("Density")
        ax1.set_title("Propensity Score Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        self.data.boxplot(column="propensity_score", by=treatment_col, ax=ax2)
        ax2.set_title("Propensity Score by Treatment Group")
        ax2.set_xlabel("Treatment Group")
        ax2.set_ylabel("Propensity Score")

        plt.tight_layout()
        return fig

    def plot_balance_assessment(
        self, covariates: list[str] | None = None, figsize: tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Plot covariate balance before and after matching."""
        if self.balance_stats is None:
            self.assess_balance(covariates)

        balance_data = []

        for covar in self.balance_stats["covariates_assessed"]:
            if covar in self.balance_stats["before_matching"]:
                before_smd = self.balance_stats["before_matching"][covar][
                    "standardized_mean_diff"
                ]
                balance_data.append(
                    {"Covariate": covar, "SMD": before_smd, "Period": "Before Matching"}
                )

                if (
                    self.balance_stats["after_matching"]
                    and covar in self.balance_stats["after_matching"]
                ):
                    after_smd = self.balance_stats["after_matching"][covar][
                        "standardized_mean_diff"
                    ]
                    balance_data.append(
                        {
                            "Covariate": covar,
                            "SMD": after_smd,
                            "Period": "After Matching",
                        }
                    )

        balance_df = pd.DataFrame(balance_data)

        fig, ax = plt.subplots(figsize=figsize)

        # Create balance plot
        if not balance_df.empty:
            sns.barplot(data=balance_df, x="SMD", y="Covariate", hue="Period", ax=ax)

            # Add reference lines for balance thresholds
            ax.axvline(
                x=0.1, color="orange", linestyle="--", alpha=0.7, label="SMD = 0.1"
            )
            ax.axvline(x=-0.1, color="orange", linestyle="--", alpha=0.7)
            ax.axvline(
                x=0.25, color="red", linestyle="--", alpha=0.7, label="SMD = 0.25"
            )
            ax.axvline(x=-0.25, color="red", linestyle="--", alpha=0.7)
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)

        ax.set_title("Covariate Balance Assessment")
        ax.set_xlabel("Standardized Mean Difference")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_treatment_effects(self, figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot treatment effect estimates with confidence intervals."""
        if self.treatment_effects is None:
            raise ValueError("Must estimate treatment effects first")

        fig, ax = plt.subplots(figsize=figsize)

        outcomes = list(self.treatment_effects.keys())
        effects = [self.treatment_effects[outcome]["ate"] for outcome in outcomes]
        ci_lower = [self.treatment_effects[outcome]["ci_lower"] for outcome in outcomes]
        ci_upper = [self.treatment_effects[outcome]["ci_upper"] for outcome in outcomes]

        # Create horizontal bar plot with error bars
        y_pos = np.arange(len(outcomes))

        bars = ax.barh(y_pos, effects, alpha=0.7)

        # Calculate error bar lengths (ensure non-negative)
        err_lower = [max(0, e - l) for e, l in zip(effects, ci_lower, strict=False)]
        err_upper = [max(0, u - e) for e, u in zip(ci_upper, effects, strict=False)]

        ax.errorbar(
            effects,
            y_pos,
            xerr=[err_lower, err_upper],
            fmt="none",
            color="black",
            capsize=5,
        )

        # Add reference line at zero
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(outcomes)
        ax.set_xlabel("Average Treatment Effect")
        ax.set_title("Treatment Effect Estimates (PSM)")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for _i, (effect, bar) in enumerate(zip(effects, bars, strict=False)):
            ax.text(
                effect + 0.01 if effect >= 0 else effect - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{effect:.3f}",
                ha="left" if effect >= 0 else "right",
                va="center",
            )

        plt.tight_layout()
        return fig

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if self.propensity_scores is None:
            return "No analysis performed yet. Please estimate propensity scores first."

        report = []
        report.append("=" * 60)
        report.append("PROPENSITY SCORE MATCHING ANALYSIS SUMMARY")
        report.append("=" * 60)

        # Propensity score model summary
        if hasattr(self, "model") and self.model is not None:
            report.append("\n1. PROPENSITY SCORE MODEL:")
            report.append("-" * 30)
            report.append("Model type: Logistic Regression")
            report.append(
                f"Propensity score range: [{self.propensity_scores.min():.3f}, {self.propensity_scores.max():.3f}]"
            )
            report.append(f"Mean propensity score: {self.propensity_scores.mean():.3f}")

        # Matching summary
        if self.matched_data is not None:
            n_treated_orig = (self.data["used_smart_assistant"] == 1).sum()
            n_treated_matched = (self.matched_data["used_smart_assistant"] == 1).sum()
            n_control_matched = (self.matched_data["used_smart_assistant"] == 0).sum()

            report.append("\n2. MATCHING RESULTS:")
            report.append("-" * 30)
            report.append(f"Original treated units: {n_treated_orig:,}")
            report.append(f"Matched treated units: {n_treated_matched:,}")
            report.append(f"Matched control units: {n_control_matched:,}")
            report.append(f"Matching rate: {n_treated_matched / n_treated_orig:.1%}")

        # Balance assessment
        if self.balance_stats:
            report.append("\n3. COVARIATE BALANCE:")
            report.append("-" * 30)

            if self.balance_stats["after_matching"]:
                balanced_vars = 0
                total_vars = 0

                for var, stats in self.balance_stats["after_matching"].items():
                    smd = abs(stats["standardized_mean_diff"])
                    total_vars += 1
                    if smd < 0.1:
                        balanced_vars += 1

                report.append(f"Variables with SMD < 0.1: {balanced_vars}/{total_vars}")
                report.append(
                    f"Balance achievement rate: {balanced_vars / total_vars:.1%}"
                )

        # Treatment effects
        if self.treatment_effects:
            report.append("\n4. TREATMENT EFFECTS:")
            report.append("-" * 30)

            for outcome, results in self.treatment_effects.items():
                ate = results["ate"]
                ci_lower = results["ci_lower"]
                ci_upper = results["ci_upper"]
                p_value = results["p_value"]

                significance = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else ""
                )

                report.append(f"{outcome}:")
                report.append(f"  ATE: {ate:.3f}{significance}")
                report.append(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                report.append(f"  P-value: {p_value:.4f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    # Helper methods
    def _prepare_features(
        self, covariates: list[str], include_interactions: bool
    ) -> np.ndarray:
        """Prepare feature matrix for propensity score estimation."""
        # Handle categorical variables
        feature_df = pd.DataFrame()

        for col in covariates:
            if col in self.data.columns:
                if self.data[col].dtype == "object":
                    # One-hot encode categorical variables
                    dummies = pd.get_dummies(
                        self.data[col], prefix=col, drop_first=True
                    )
                    feature_df = pd.concat([feature_df, dummies], axis=1)
                else:
                    # Keep numeric variables as-is
                    feature_df[col] = self.data[col]

        # Add interactions if requested
        if include_interactions:
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    feature_df[f"{col1}_x_{col2}"] = feature_df[col1] * feature_df[col2]

        # Scale features
        return self.scaler.fit_transform(feature_df)

    def _get_feature_names(
        self, covariates: list[str], include_interactions: bool
    ) -> list[str]:
        """Get feature names after preprocessing."""
        feature_names = []

        for col in covariates:
            if col in self.data.columns:
                if self.data[col].dtype == "object":
                    unique_vals = self.data[col].unique()
                    for val in sorted(unique_vals)[
                        1:
                    ]:  # Skip first for drop_first=True
                        feature_names.append(f"{col}_{val}")
                else:
                    feature_names.append(col)

        if include_interactions:
            numeric_features = [
                name
                for name in feature_names
                if not any(cat in name for cat in ["_", "bracket", "type"])
            ]
            for i, feat1 in enumerate(numeric_features):
                for feat2 in numeric_features[i + 1 :]:
                    feature_names.append(f"{feat1}_x_{feat2}")

        return feature_names

    def _assess_common_support(
        self, treatment: pd.Series, propensity_scores: np.ndarray
    ) -> dict[str, Any]:
        """Assess common support region."""
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]

        min_treated = treated_ps.min()
        max_treated = treated_ps.max()
        min_control = control_ps.min()
        max_control = control_ps.max()

        common_min = max(min_treated, min_control)
        common_max = min(max_treated, max_control)

        return {
            "common_support_range": [common_min, common_max],
            "treated_range": [min_treated, max_treated],
            "control_range": [min_control, max_control],
            "overlap_ratio": (common_max - common_min) / (max_treated - min_treated)
            if max_treated > min_treated
            else 0,
        }

    def _nearest_neighbor_matching(
        self,
        treated_indices: pd.Index,
        control_indices: pd.Index,
        treated_ps: np.ndarray,
        control_ps: np.ndarray,
        caliper: float | None,
        replacement: bool,
        ratio: int,
    ) -> dict[int, list[int]]:
        """Perform nearest neighbor matching."""
        # Use sklearn's NearestNeighbors for efficient matching
        nn = NearestNeighbors(n_neighbors=ratio, metric="euclidean")
        nn.fit(control_ps.reshape(-1, 1))

        matches = {}
        used_controls = set()

        for i, treated_idx in enumerate(treated_indices):
            treated_score = (
                treated_ps.iloc[i] if hasattr(treated_ps, "iloc") else treated_ps[i]
            )

            # Find nearest neighbors
            distances, indices = nn.kneighbors([[treated_score]])

            matched_controls = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                control_idx = control_indices[idx]

                # Check caliper constraint
                if caliper is not None and dist > caliper:
                    continue

                # Check replacement constraint
                if not replacement and control_idx in used_controls:
                    continue

                matched_controls.append(control_idx)
                if not replacement:
                    used_controls.add(control_idx)

            matches[treated_idx] = matched_controls

        return matches

    def _caliper_matching(
        self,
        treated_indices: pd.Index,
        control_indices: pd.Index,
        treated_ps: np.ndarray,
        control_ps: np.ndarray,
        caliper: float,
    ) -> dict[int, list[int]]:
        """Perform caliper matching."""
        matches = {}

        for i, treated_idx in enumerate(treated_indices):
            treated_score = (
                treated_ps.iloc[i] if hasattr(treated_ps, "iloc") else treated_ps[i]
            )

            # Find all controls within caliper
            distances = np.abs(control_ps - treated_score)
            within_caliper = distances <= caliper

            matched_controls = control_indices[within_caliper].tolist()
            matches[treated_idx] = matched_controls

        return matches

    def _calculate_balance_stats(
        self, data: pd.DataFrame, covariates: list[str], treatment_col: str
    ) -> dict[str, dict[str, float]]:
        """Calculate balance statistics for covariates."""
        balance_stats = {}

        treated = data[data[treatment_col] == 1]
        control = data[data[treatment_col] == 0]

        for covar in covariates:
            if covar not in data.columns:
                continue

            # Skip non-informative columns
            if covar in ["user_id", "id"] or covar == treatment_col:
                continue

            if data[covar].dtype == "object":
                # For categorical variables, use standardized difference of proportions
                treated_props = treated[covar].value_counts(normalize=True)
                control_props = control[covar].value_counts(normalize=True)

                # Calculate overall standardized mean difference for categorical
                smd = 0
                for category in set(treated_props.index) | set(control_props.index):
                    p_t = treated_props.get(category, 0)
                    p_c = control_props.get(category, 0)
                    pooled_var = (p_t * (1 - p_t) + p_c * (1 - p_c)) / 2
                    if pooled_var > 0:
                        smd += abs(p_t - p_c) / np.sqrt(pooled_var)

                balance_stats[covar] = {
                    "standardized_mean_diff": smd,
                    "treated_mean": np.nan,  # Not applicable for categorical
                    "control_mean": np.nan,
                    "p_value": np.nan,
                }
            else:
                # For numeric variables
                treated_vals = treated[covar].dropna()
                control_vals = control[covar].dropna()

                if len(treated_vals) == 0 or len(control_vals) == 0:
                    continue

                # Standardized mean difference
                mean_diff = treated_vals.mean() - control_vals.mean()
                pooled_std = np.sqrt((treated_vals.var() + control_vals.var()) / 2)
                smd = mean_diff / pooled_std if pooled_std > 0 else 0

                # T-test with proper error handling
                try:
                    if len(treated_vals) > 1 and len(control_vals) > 1:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="Precision loss occurred in moment calculation",
                                category=RuntimeWarning,
                            )
                            _, p_value = stats.ttest_ind(treated_vals, control_vals)

                            # Handle edge cases where p-value might be invalid
                            if np.isnan(p_value) or np.isinf(p_value):
                                p_value = 1.0  # Conservative p-value when test fails
                    else:
                        p_value = 1.0  # Conservative p-value with insufficient data

                except (ValueError, RuntimeError, TypeError) as e:
                    # Only use conservative fallback for actual errors
                    print(
                        f"Warning: T-test failed with error {e}, using conservative p-value"
                    )
                    p_value = 1.0  # Conservative p-value when test fails

                balance_stats[covar] = {
                    "standardized_mean_diff": smd,
                    "treated_mean": treated_vals.mean(),
                    "control_mean": control_vals.mean(),
                    "p_value": p_value,
                }

        return balance_stats

    def _simple_difference_estimator(
        self, data: pd.DataFrame, outcome_col: str, treatment_col: str
    ) -> dict[str, float]:
        """Estimate treatment effect using simple difference of means."""
        treated_outcome = data[data[treatment_col] == 1][outcome_col].dropna()
        control_outcome = data[data[treatment_col] == 0][outcome_col].dropna()

        if len(treated_outcome) == 0 or len(control_outcome) == 0:
            return {
                "ate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "n_treated": len(treated_outcome),
                "n_control": len(control_outcome),
            }

        # Convert boolean to numeric if needed
        if treated_outcome.dtype == bool:
            treated_outcome = treated_outcome.astype(int)
        if control_outcome.dtype == bool:
            control_outcome = control_outcome.astype(int)

        # Calculate ATE
        ate = treated_outcome.mean() - control_outcome.mean()

        # Calculate standard error and confidence interval
        se_treated = treated_outcome.std() / np.sqrt(len(treated_outcome))
        se_control = control_outcome.std() / np.sqrt(len(control_outcome))
        se_ate = np.sqrt(se_treated**2 + se_control**2)

        # 95% confidence interval
        ci_lower = ate - 1.96 * se_ate
        ci_upper = ate + 1.96 * se_ate

        # Two-sample t-test
        try:
            if len(treated_outcome) > 1 and len(control_outcome) > 1:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Precision loss occurred in moment calculation",
                        category=RuntimeWarning,
                    )
                    _, p_value = stats.ttest_ind(treated_outcome, control_outcome)

                    # Handle edge cases where p-value might be invalid
                    if np.isnan(p_value) or np.isinf(p_value):
                        # For binary outcomes, use a proportion test instead
                        if data[outcome_col].dtype == bool or set(
                            data[outcome_col].unique()
                        ).issubset({0, 1}):
                            p_value = self._proportion_test(
                                treated_outcome, control_outcome
                            )
                        else:
                            p_value = 1.0  # Conservative fallback
            else:
                p_value = 1.0  # Conservative p-value with insufficient data

        except (ValueError, RuntimeError, TypeError):
            # For binary outcomes, try proportion test as fallback
            if data[outcome_col].dtype == bool or set(
                data[outcome_col].unique()
            ).issubset({0, 1}):
                p_value = self._proportion_test(treated_outcome, control_outcome)
            else:
                p_value = 1.0  # Conservative p-value when test fails

        return {
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "standard_error": se_ate,
            "n_treated": len(treated_outcome),
            "n_control": len(control_outcome),
            "treated_mean": treated_outcome.mean(),
            "control_mean": control_outcome.mean(),
        }

    def _proportion_test(self, treated: pd.Series, control: pd.Series) -> float:
        """Perform a two-proportion z-test for binary outcomes."""
        try:
            # Convert to numeric if boolean
            if treated.dtype == bool:
                treated = treated.astype(int)
            if control.dtype == bool:
                control = control.astype(int)

            n1, n2 = len(treated), len(control)
            p1, p2 = treated.mean(), control.mean()

            # Pooled proportion
            p_pooled = (treated.sum() + control.sum()) / (n1 + n2)

            # Standard error for difference in proportions
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

            if se == 0:
                return 1.0  # No variation, conservative p-value

            # Z-statistic
            z_stat = (p1 - p2) / se

            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            return p_value

        except (ValueError, ZeroDivisionError):
            return 1.0  # Conservative fallback


def load_and_analyze_psm(
    file_path: str,
    treatment_col: str = "used_smart_assistant",
    outcome_cols: str | list[str] = "filed_2024",
    covariates: list[str] | None = None,
    matching_method: str = "nearest_neighbor",
    caliper: float | None = None,
) -> PropensityScoreMatching:
    """
    Convenience function to load data and perform complete PSM analysis.

    Args:
        file_path: Path to CSV data file
        treatment_col: Name of treatment variable
        outcome_cols: Outcome variable(s) to analyze
        covariates: List of covariate column names
        matching_method: Matching method to use
        caliper: Caliper for matching

    Returns:
        Fitted PropensityScoreMatching object
    """
    # Load data
    data = pd.read_csv(file_path)

    # Initialize PSM
    psm = PropensityScoreMatching(data)

    # Estimate propensity scores
    psm.estimate_propensity_scores(treatment_col=treatment_col, covariates=covariates)

    # Perform matching
    psm.perform_matching(
        method=matching_method, caliper=caliper, treatment_col=treatment_col
    )

    # Assess balance
    psm.assess_balance(covariates=covariates, treatment_col=treatment_col)

    # Estimate treatment effects
    psm.estimate_treatment_effects(
        outcome_cols=outcome_cols, treatment_col=treatment_col
    )

    return psm
