"""
Double Machine Learning (DML) for Causal Inference

This module implements Double Machine Learning methods for estimating causal effects
in observational data using machine learning models with cross-fitting to avoid
bias from overfitting.

Key Features:
- Multiple ML models (Random Forest, Gradient Boosting, Logistic/Linear Regression)
- Cross-fitting with configurable folds
- Robust treatment effect estimation with confidence intervals
- Comprehensive diagnostics and model performance metrics
- Visualization tools for model assessment and results
- Proper handling of binary and continuous outcomes

References:
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment and structural parameters"
- Athey & Imbens (2019). "Machine Learning Methods for Estimating Heterogeneous Causal Effects"
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class DoubleMachineLearning:
    """
    Double Machine Learning for causal inference.

    Uses machine learning models to estimate both outcome and treatment propensity,
    with cross-fitting to obtain unbiased treatment effect estimates.
    """

    def __init__(self, data: pd.DataFrame, random_state: int = 42):
        """
        Initialize DML estimator.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing treatment, outcome, and covariates
        random_state : int
            Random seed for reproducibility
        """
        self.data = data.copy()
        self.random_state = random_state
        self.n_folds = 2  # Will be set in estimate_treatment_effects
        self.outcome_models = {}
        self.treatment_models = {}
        self.treatment_effects = {}
        self.model_performance = {}
        self.residuals = {}
        self.fitted = False

        # Set random seeds
        np.random.seed(random_state)

    def get_available_models(self, model_type: str = "regression") -> dict[str, Any]:
        """
        Get available ML models for outcome or treatment estimation.

        Parameters:
        -----------
        model_type : str
            Either 'regression' for continuous outcomes or 'classification' for binary treatments

        Returns:
        --------
        Dict[str, Any]
            Dictionary of model names and their sklearn instances
        """
        if model_type == "regression":
            return {
                "random_forest": RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                "linear_regression": LinearRegression(),
                "ridge": Ridge(alpha=1.0, random_state=self.random_state),
            }
        elif model_type == "classification":
            return {
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                "gradient_boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                "logistic_regression": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                "ridge_classifier": RidgeClassifier(
                    alpha=1.0, random_state=self.random_state
                ),
            }
        else:
            raise ValueError("model_type must be 'regression' or 'classification'")

    def _prepare_data(
        self, outcome_col: str, treatment_col: str, covariates: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for DML estimation.

        Parameters:
        -----------
        outcome_col : str
            Name of outcome column
        treatment_col : str
            Name of treatment column
        covariates : List[str]
            List of covariate column names

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays for outcome (Y), treatment (D), and covariates (X)
        """
        # Remove rows with missing values
        analysis_cols = [outcome_col, treatment_col] + covariates
        clean_data = self.data[analysis_cols].dropna()

        if len(clean_data) == 0:
            raise ValueError("No complete cases found after removing missing values")

        Y = clean_data[outcome_col].values
        D = clean_data[treatment_col].values
        X = clean_data[covariates].values

        # Convert boolean outcomes to numeric
        if Y.dtype == bool:
            Y = Y.astype(int)
        if D.dtype == bool:
            D = D.astype(int)

        return Y, D, X

    def _evaluate_model_performance(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_type: str,
    ) -> dict[str, float]:
        """
        Evaluate model performance on test set.

        Parameters:
        -----------
        model : sklearn model
            Fitted model to evaluate
        X_train, X_test : np.ndarray
            Training and test features
        y_train, y_test : np.ndarray
            Training and test targets
        model_type : str
            Either 'regression' or 'classification'

        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        y_pred = model.predict(X_test)

        if model_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {"mse": mse, "rmse": np.sqrt(mse), "r2": r2}
        else:  # classification
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    y_pred_proba = proba[:, 1]
                    if len(np.unique(y_test)) > 1:  # Need at least 2 classes for AUC
                        auc = roc_auc_score(y_test, y_pred_proba)
                    else:
                        auc = np.nan
                else:
                    auc = np.nan
            else:
                auc = np.nan

            accuracy = accuracy_score(y_test, y_pred)
            return {"accuracy": accuracy, "auc": auc}

    def estimate_treatment_effects(
        self,
        outcome_col: str,
        treatment_col: str,
        covariates: list[str],
        outcome_model: str = "random_forest",
        treatment_model: str = "random_forest",
        n_folds: int = 2,
        scale_features: bool = True,
    ) -> dict[str, Any]:
        """
        Estimate treatment effects using Double Machine Learning.

        Parameters:
        -----------
        outcome_col : str
            Name of outcome column
        treatment_col : str
            Name of treatment column
        covariates : List[str]
            List of covariate column names
        outcome_model : str
            ML model for outcome prediction ('random_forest', 'gradient_boosting',
            'linear_regression', 'ridge')
        treatment_model : str
            ML model for treatment prediction ('random_forest', 'gradient_boosting',
            'logistic_regression', 'ridge_classifier')
        n_folds : int
            Number of folds for cross-fitting
        scale_features : bool
            Whether to standardize features

        Returns:
        --------
        Dict[str, Any]
            Treatment effect estimates and diagnostics
        """
        self.n_folds = n_folds

        # Prepare data
        Y, D, X = self._prepare_data(outcome_col, treatment_col, covariates)
        n_samples = len(Y)

        # Determine if outcome is binary
        outcome_is_binary = len(np.unique(Y)) == 2

        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Get models
        outcome_model_type = "classification" if outcome_is_binary else "regression"
        outcome_models = self.get_available_models(outcome_model_type)
        treatment_models = self.get_available_models("classification")

        if outcome_model not in outcome_models:
            raise ValueError(
                f"Unknown outcome model: {outcome_model}. Available: {list(outcome_models.keys())}"
            )
        if treatment_model not in treatment_models:
            raise ValueError(
                f"Unknown treatment model: {treatment_model}. Available: {list(treatment_models.keys())}"
            )

        # Initialize cross-fitting
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        # Store residuals and predictions for each fold
        outcome_residuals = np.zeros(n_samples)
        treatment_residuals = np.zeros(n_samples)
        fold_performance = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]

            # Fit outcome model on training data, predict on test data
            outcome_model_instance = outcome_models[outcome_model]
            outcome_model_instance.fit(X_train, Y_train)

            if outcome_is_binary:
                if hasattr(outcome_model_instance, 'predict_proba'):
                    proba = outcome_model_instance.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        Y_pred = proba[:, 1]
                    else:
                        # Only one class in training data
                        Y_pred = np.full(len(X_test), proba[0, 0])
                else:
                    # Model doesn't have predict_proba, use predict
                    Y_pred = outcome_model_instance.predict(X_test).astype(float)
            else:
                Y_pred = outcome_model_instance.predict(X_test)

            # Fit treatment model on training data, predict on test data
            treatment_model_instance = treatment_models[treatment_model]
            treatment_model_instance.fit(X_train, D_train)

            if hasattr(treatment_model_instance, 'predict_proba'):
                proba = treatment_model_instance.predict_proba(X_test)
                if proba.shape[1] == 2:
                    D_pred = proba[:, 1]
                else:
                    # Only one class in training data
                    D_pred = np.full(len(X_test), proba[0, 0])
            else:
                # Model doesn't have predict_proba, use predict
                D_pred = treatment_model_instance.predict(X_test).astype(float)

            # Calculate residuals
            outcome_residuals[test_idx] = Y_test - Y_pred
            treatment_residuals[test_idx] = D_test - D_pred

            # Evaluate model performance
            try:
                outcome_perf = self._evaluate_model_performance(
                    outcome_model_instance,
                    X_train,
                    X_test,
                    Y_train,
                    Y_test,
                    outcome_model_type,
                )
            except Exception:
                # If performance evaluation fails, use dummy values
                outcome_perf = (
                    {"r2": np.nan, "mse": np.nan, "rmse": np.nan}
                    if not outcome_is_binary
                    else {"accuracy": np.nan, "auc": np.nan}
                )

            try:
                treatment_perf = self._evaluate_model_performance(
                    treatment_model_instance,
                    X_train,
                    X_test,
                    D_train,
                    D_test,
                    "classification",
                )
            except Exception:
                # If performance evaluation fails, use dummy values
                treatment_perf = {"accuracy": np.nan, "auc": np.nan}

            fold_performance.append(
                {
                    "fold": fold_idx,
                    "outcome_performance": outcome_perf,
                    "treatment_performance": treatment_perf,
                }
            )

        # Estimate ATE using orthogonal moments
        # ATE = E[outcome_residuals * treatment_residuals] / E[treatment_residuals^2]
        numerator = np.mean(outcome_residuals * treatment_residuals)
        denominator = np.mean(treatment_residuals**2)

        if abs(denominator) < 1e-10:
            warnings.warn(
                "Treatment residuals have very low variance. Results may be unreliable.",
                stacklevel=2
            )
            ate = np.nan
            se = np.nan
        else:
            ate = numerator / denominator

            # Calculate standard error using influence function
            influence_function = (
                outcome_residuals * treatment_residuals - ate * treatment_residuals**2
            ) / denominator
            se = np.sqrt(np.var(influence_function) / n_samples)

        # Calculate confidence interval
        t_stat = stats.t.ppf(0.975, n_samples - 1)
        ci_lower = ate - t_stat * se
        ci_upper = ate + t_stat * se

        # Calculate p-value
        if not np.isnan(se) and se > 0:
            p_value = 2 * (1 - stats.t.cdf(abs(ate / se), n_samples - 1))
        else:
            p_value = np.nan

        # Store results
        results = {
            "ate": ate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "n_samples": n_samples,
            "outcome_model": outcome_model,
            "treatment_model": treatment_model,
            "n_folds": n_folds,
            "outcome_is_binary": outcome_is_binary,
            "fold_performance": fold_performance,
        }

        # Store for later use
        self.treatment_effects[outcome_col] = results
        self.residuals[outcome_col] = {
            "outcome_residuals": outcome_residuals,
            "treatment_residuals": treatment_residuals,
        }
        self.fitted = True

        return results

    def estimate_multiple_outcomes(
        self,
        outcome_cols: list[str],
        treatment_col: str,
        covariates: list[str],
        outcome_model: str = "random_forest",
        treatment_model: str = "random_forest",
        n_folds: int = 2,
        scale_features: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Estimate treatment effects for multiple outcomes.

        Parameters:
        -----------
        outcome_cols : List[str]
            List of outcome column names
        treatment_col : str
            Name of treatment column
        covariates : List[str]
            List of covariate column names
        outcome_model : str
            ML model for outcome prediction
        treatment_model : str
            ML model for treatment prediction
        n_folds : int
            Number of folds for cross-fitting
        scale_features : bool
            Whether to standardize features

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Treatment effect estimates for each outcome
        """
        results = {}
        for outcome_col in outcome_cols:
            try:
                results[outcome_col] = self.estimate_treatment_effects(
                    outcome_col=outcome_col,
                    treatment_col=treatment_col,
                    covariates=covariates,
                    outcome_model=outcome_model,
                    treatment_model=treatment_model,
                    n_folds=n_folds,
                    scale_features=scale_features,
                )
            except Exception as e:
                warnings.warn(f"Failed to estimate effects for {outcome_col}: {str(e)}", stacklevel=2)
                results[outcome_col] = {
                    "ate": np.nan,
                    "se": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "error": str(e),
                }

        return results

    def compare_models(
        self,
        outcome_col: str,
        treatment_col: str,
        covariates: list[str],
        n_folds: int = 2,
        scale_features: bool = True,
    ) -> pd.DataFrame:
        """
        Compare different ML models for DML estimation.

        Parameters:
        -----------
        outcome_col : str
            Name of outcome column
        treatment_col : str
            Name of treatment column
        covariates : List[str]
            List of covariate column names
        n_folds : int
            Number of folds for cross-fitting
        scale_features : bool
            Whether to standardize features

        Returns:
        --------
        pd.DataFrame
            Comparison of model performance and treatment effect estimates
        """
        # Determine if outcome is binary
        Y, _, _ = self._prepare_data(outcome_col, treatment_col, covariates)
        outcome_is_binary = len(np.unique(Y)) == 2

        # Get available models
        outcome_model_type = "classification" if outcome_is_binary else "regression"
        outcome_models = list(self.get_available_models(outcome_model_type).keys())
        treatment_models = list(self.get_available_models("classification").keys())

        results = []

        for outcome_model in outcome_models:
            for treatment_model in treatment_models:
                try:
                    result = self.estimate_treatment_effects(
                        outcome_col=outcome_col,
                        treatment_col=treatment_col,
                        covariates=covariates,
                        outcome_model=outcome_model,
                        treatment_model=treatment_model,
                        n_folds=n_folds,
                        scale_features=scale_features,
                    )

                    results.append(
                        {
                            "outcome_model": outcome_model,
                            "treatment_model": treatment_model,
                            "ate": result["ate"],
                            "se": result["se"],
                            "p_value": result["p_value"],
                            "ci_lower": result["ci_lower"],
                            "ci_upper": result["ci_upper"],
                        }
                    )

                except Exception as e:
                    warnings.warn(
                        f"Failed combination {outcome_model} + {treatment_model}: {str(e)}",
                        stacklevel=2
                    )
                    continue

        return pd.DataFrame(results)

    def plot_residuals(
        self, outcome_col: str, figsize: tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Plot residuals from outcome and treatment models.

        Parameters:
        -----------
        outcome_col : str
            Name of outcome column to plot residuals for
        figsize : Tuple[int, int]
            Figure size for the plot

        Returns:
        --------
        plt.Figure
            Figure object containing the residual plots
        """
        if outcome_col not in self.residuals:
            raise ValueError(
                f"No residuals found for {outcome_col}. Run estimate_treatment_effects first."
            )

        residuals = self.residuals[outcome_col]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Outcome residuals
        axes[0].hist(
            residuals["outcome_residuals"], bins=30, alpha=0.7, edgecolor="black"
        )
        axes[0].set_title("Outcome Model Residuals")
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)

        # Treatment residuals
        axes[1].hist(
            residuals["treatment_residuals"], bins=30, alpha=0.7, edgecolor="black"
        )
        axes[1].set_title("Treatment Model Residuals")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout()
        return fig

    def plot_treatment_effects(self, figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot treatment effect estimates with confidence intervals.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size for the plot

        Returns:
        --------
        plt.Figure
            Figure object containing the treatment effect plot
        """
        if not self.treatment_effects:
            raise ValueError(
                "No treatment effects estimated. Run estimate_treatment_effects first."
            )

        # Prepare data for plotting
        outcomes = []
        ates = []
        ci_lowers = []
        ci_uppers = []
        p_values = []

        for outcome, results in self.treatment_effects.items():
            outcomes.append(outcome)
            ates.append(results["ate"])
            ci_lowers.append(results["ci_lower"])
            ci_uppers.append(results["ci_upper"])
            p_values.append(results["p_value"])

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(outcomes))

        # Plot point estimates
        colors = ["red" if p < 0.05 else "blue" for p in p_values]
        ax.scatter(ates, y_pos, color=colors, s=100, zorder=3)

        # Plot confidence intervals
        for i, (_ate, ci_lower, ci_upper) in enumerate(
            zip(ates, ci_lowers, ci_uppers, strict=False)
        ):
            ax.plot(
                [ci_lower, ci_upper], [i, i], color=colors[i], linewidth=2, alpha=0.7
            )

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(outcomes)
        ax.set_xlabel("Average Treatment Effect")
        ax.set_title("DML Treatment Effect Estimates with 95% Confidence Intervals")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="p < 0.05",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="p ≥ 0.05",
            ),
        ]
        ax.legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        return fig

    def generate_summary_report(self, outcome_col: str | None = None) -> str:
        """
        Generate a summary report of DML results.

        Parameters:
        -----------
        outcome_col : Optional[str]
            Specific outcome to report on. If None, reports on all outcomes.

        Returns:
        --------
        str
            Formatted summary report
        """
        if not self.treatment_effects:
            return (
                "No treatment effects estimated. Run estimate_treatment_effects first."
            )

        if outcome_col and outcome_col not in self.treatment_effects:
            return f"No results found for outcome: {outcome_col}"

        outcomes_to_report = (
            [outcome_col] if outcome_col else list(self.treatment_effects.keys())
        )

        report = "=" * 50 + "\n"
        report += "DOUBLE MACHINE LEARNING RESULTS\n"
        report += "=" * 50 + "\n\n"

        for outcome in outcomes_to_report:
            results = self.treatment_effects[outcome]

            report += f"Outcome: {outcome}\n"
            report += "-" * 30 + "\n"
            report += f"Average Treatment Effect: {results['ate']:.4f}\n"
            report += f"Standard Error: {results['se']:.4f}\n"
            report += f"95% Confidence Interval: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]\n"
            report += f"P-value: {results['p_value']:.4f}\n"

            if results["p_value"] < 0.05:
                report += "*** STATISTICALLY SIGNIFICANT ***\n"
            else:
                report += "Not statistically significant\n"

            report += "\nModel Details:\n"
            report += f"  Outcome Model: {results['outcome_model']}\n"
            report += f"  Treatment Model: {results['treatment_model']}\n"
            report += f"  Sample Size: {results['n_samples']}\n"
            report += f"  Cross-fitting Folds: {results['n_folds']}\n"
            report += f"  Binary Outcome: {results['outcome_is_binary']}\n"

            # Add model performance summary
            if "fold_performance" in results:
                avg_outcome_perf = {}
                avg_treatment_perf = {}

                for fold_data in results["fold_performance"]:
                    for metric, value in fold_data["outcome_performance"].items():
                        if metric not in avg_outcome_perf:
                            avg_outcome_perf[metric] = []
                        avg_outcome_perf[metric].append(value)

                    for metric, value in fold_data["treatment_performance"].items():
                        if metric not in avg_treatment_perf:
                            avg_treatment_perf[metric] = []
                        avg_treatment_perf[metric].append(value)

                report += "\nModel Performance (Cross-Validation):\n"
                report += "  Outcome Model Performance:\n"
                for metric, values in avg_outcome_perf.items():
                    if not any(np.isnan(values)):
                        report += f"    {metric.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})\n"

                report += "  Treatment Model Performance:\n"
                for metric, values in avg_treatment_perf.items():
                    if not any(np.isnan(values)):
                        report += f"    {metric.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})\n"

            report += "\n" + "=" * 50 + "\n\n"

        return report


def load_and_analyze_dml(
    file_path: str,
    outcome_cols: str | list[str],
    treatment_col: str,
    covariates: list[str],
    outcome_model: str = "random_forest",
    treatment_model: str = "random_forest",
    n_folds: int = 2,
    scale_features: bool = True,
) -> DoubleMachineLearning:
    """
    Convenience function to load data and perform DML analysis.

    Parameters:
    -----------
    file_path : str
        Path to CSV file containing the data
    outcome_cols : Union[str, List[str]]
        Outcome column name(s)
    treatment_col : str
        Treatment column name
    covariates : List[str]
        List of covariate column names
    outcome_model : str
        ML model for outcome prediction
    treatment_model : str
        ML model for treatment prediction
    n_folds : int
        Number of folds for cross-fitting
    scale_features : bool
        Whether to standardize features

    Returns:
    --------
    DoubleMachineLearning
        Fitted DML object with results
    """
    # Load data
    df = pd.read_csv(file_path)

    # Initialize DML
    dml = DoubleMachineLearning(df)

    # Estimate treatment effects
    if isinstance(outcome_cols, str):
        dml.estimate_treatment_effects(
            outcome_col=outcome_cols,
            treatment_col=treatment_col,
            covariates=covariates,
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            n_folds=n_folds,
            scale_features=scale_features,
        )
    else:
        dml.estimate_multiple_outcomes(
            outcome_cols=outcome_cols,
            treatment_col=treatment_col,
            covariates=covariates,
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            n_folds=n_folds,
            scale_features=scale_features,
        )

    return dml
