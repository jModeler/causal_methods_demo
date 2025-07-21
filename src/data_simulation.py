"""
Data simulation module for generating realistic tax software user data.

This module creates synthetic data for demonstrating causal inference methods
in the context of analyzing the impact of a "Smart Filing Assistant" feature
on user conversion and engagement.

The simulation parameters are now configurable via YAML files for better
maintainability and experimentation.
"""

import pandas as pd
import numpy as np
from faker import Faker
import yaml
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime, timedelta
import os


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    import copy
    
    merged = copy.deepcopy(base_config)
    
    def recursive_merge(base_dict, override_dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                recursive_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    recursive_merge(merged, override_config)
    return merged


def load_config(config_path: str = "config/simulation_config.yaml") -> Dict:
    """
    Load simulation configuration from YAML file with inheritance support.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing simulation parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # If this is not the base config, merge with base config
    if 'simulation_config.yaml' not in config_path:
        base_config_path = os.path.join(os.path.dirname(config_path), 'simulation_config.yaml')
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as base_file:
                base_config = yaml.safe_load(base_file)
            config = merge_configs(base_config, config)
    
    return config


class TaxSoftwareDataSimulator:
    """Simulates realistic tax software user data with causal relationships."""
    
    def __init__(self, n_users: int = None, config_path: str = "config/simulation_config.yaml"):
        """
        Initialize the data simulator.
        
        Args:
            n_users: Number of users to generate (overrides config if specified)
            config_path: Path to the YAML configuration file
        """
        self.config = load_config(config_path)
        
        # Set number of users (parameter overrides config)
        self.n_users = n_users if n_users is not None else self.config['simulation']['default_n_users']
        
        # Set random seeds for reproducibility
        seed = self.config['simulation']['random_seed']
        np.random.seed(seed)
        random.seed(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)
        
    def generate_user_demographics(self) -> pd.DataFrame:
        """Generate user demographic and metadata features."""
        
        # Get demographic config
        demo_config = self.config['demographics']
        tech_config = self.config['tech_savviness']
        
        # Income brackets
        income_brackets = demo_config['income_brackets']['values']
        income_weights = demo_config['income_brackets']['weights']
        
        # Device types
        device_types = demo_config['device_types']['values']
        device_weights = demo_config['device_types']['weights']
        
        # User types
        user_types = demo_config['user_types']['values']
        user_type_weights = demo_config['user_types']['weights']
        
        # Geographic regions
        regions = demo_config['regions']['values']
        region_weights = demo_config['regions']['weights']
        
        data = []
        for i in range(self.n_users):
            user_id = f"user_{i:06d}"
            income_bracket = np.random.choice(income_brackets, p=income_weights)
            device_type = np.random.choice(device_types, p=device_weights)
            user_type = np.random.choice(user_types, p=user_type_weights)
            region = np.random.choice(regions, p=region_weights)
            
            # Age correlated with tech-savviness
            age_params = demo_config['age']
            age = int(np.random.normal(age_params['mean'], age_params['std']))
            age = max(age_params['min_age'], min(age_params['max_age'], age))
            
            # Tech-savviness score (hidden confounders)
            tech_base = tech_config['base_score']
            
            # Age-based adjustments
            age_adj = tech_config['age_adjustments']
            if age < age_adj['young_threshold']:
                tech_base += age_adj['young_boost']
            elif age > age_adj['old_threshold']:
                tech_base += age_adj['old_penalty']  # Note: this is negative
                
            # Income-based adjustments
            income_adj = tech_config['income_adjustments']
            if income_bracket in income_adj['high_income_brackets']:
                tech_base += income_adj['high_income_boost']
            elif income_bracket in income_adj['low_income_brackets']:
                tech_base += income_adj['low_income_penalty']  # Note: this is negative
                
            # Regional adjustments
            region_adj = tech_config['region_adjustments']
            if region == 'West':
                tech_base += region_adj['west_boost']
            elif region in ['Midwest', 'South']:
                tech_base += region_adj['midwest_south_penalty']  # Note: this is negative
                
            tech_savviness = max(tech_config['min_score'], 
                               min(tech_config['max_score'], 
                                   int(np.random.normal(tech_base, tech_config['std']))))
            
            data.append({
                'user_id': user_id,
                'age': age,
                'income_bracket': income_bracket,
                'device_type': device_type,
                'user_type': user_type,
                'region': region,
                'tech_savviness': tech_savviness
            })
            
        return pd.DataFrame(data)
    
    def generate_2023_baseline_data(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate 2023 baseline behavior data (pre-treatment)."""
        
        baseline_config = self.config['baseline_2023']
        baseline_data = []
        
        for _, user in users_df.iterrows():
            # Filing behavior in 2023
            filing_config = baseline_config['filing']
            filed_2023 = np.random.random() < self._get_filing_probability(user, filing_config)
            
            if filed_2023:
                # Time to complete filing (minutes)
                time_config = baseline_config['time_to_complete']
                base_time = time_config['base_time']
                
                if user['user_type'] == 'returning':
                    base_time += time_config['returning_user_reduction']  # negative value
                if user['tech_savviness'] > 70:
                    base_time += time_config['high_tech_reduction']  # negative value
                elif user['tech_savviness'] < 30:
                    base_time += time_config['low_tech_penalty']
                    
                time_to_complete_2023 = max(time_config['min_time'], 
                                          int(np.random.normal(base_time, time_config['std'])))
                
                # Number of sessions
                session_config = baseline_config['sessions']
                base_sessions = session_config['base_sessions']
                
                if user['device_type'] == 'mobile':
                    base_sessions += session_config['mobile_penalty']
                if user['tech_savviness'] < 40:
                    base_sessions += session_config['low_tech_penalty']
                    
                sessions_2023 = max(1, int(np.random.poisson(base_sessions)))
                
                # Support tickets
                support_config = baseline_config['support_tickets']
                support_prob = support_config['base_rate']
                
                if user['tech_savviness'] < 30:
                    support_prob += support_config['low_tech_boost']
                if user['age'] > 65:
                    support_prob += support_config['elderly_boost']
                    
                support_tickets_2023 = np.random.poisson(support_prob)
            else:
                time_to_complete_2023 = 0
                sessions_2023 = 0
                support_tickets_2023 = 0
            
            # Early login behavior (predictor of feature adoption)
            early_config = self.config['early_login']
            early_login_prob = early_config['base_probability']
            
            if user['tech_savviness'] > 60:
                early_login_prob += early_config['high_tech_boost']
            if user['user_type'] == 'returning':
                early_login_prob += early_config['returning_user_boost']
                
            early_login_2024 = np.random.random() < early_login_prob
            
            baseline_data.append({
                'user_id': user['user_id'],
                'filed_2023': filed_2023,
                'time_to_complete_2023': time_to_complete_2023,
                'sessions_2023': sessions_2023,
                'support_tickets_2023': support_tickets_2023,
                'early_login_2024': early_login_2024
            })
            
        return pd.DataFrame(baseline_data)
    
    def generate_treatment_assignment(self, users_df: pd.DataFrame, 
                                    baseline_df: pd.DataFrame) -> pd.DataFrame:
        """Generate Smart Filing Assistant usage (treatment assignment)."""
        
        treatment_config = self.config['treatment']
        treatment_data = []
        
        for _, user in users_df.iterrows():
            baseline_user = baseline_df[baseline_df['user_id'] == user['user_id']].iloc[0]
            
            # Probability of using Smart Filing Assistant
            use_prob = treatment_config['base_adoption_rate']
            
            # Tech-savviness is main driver
            tech_effects = treatment_config['tech_effects']
            if user['tech_savviness'] > tech_effects['high_threshold']:
                use_prob += tech_effects['high_boost']
            elif user['tech_savviness'] > tech_effects['medium_threshold']:
                use_prob += tech_effects['medium_boost']
            elif user['tech_savviness'] < tech_effects['low_threshold']:
                use_prob += tech_effects['low_penalty']  # negative value
            
            # Age effects
            age_effects = treatment_config['age_effects']
            if user['age'] < age_effects['young_threshold']:
                use_prob += age_effects['young_boost']
            elif user['age'] > age_effects['old_threshold']:
                use_prob += age_effects['old_penalty']  # negative value
                
            # Device effects
            device_effects = treatment_config['device_effects']
            if user['device_type'] == 'mobile':
                use_prob += device_effects['mobile_boost']
            elif user['device_type'] == 'tablet':
                use_prob += device_effects['tablet_penalty']  # negative value
                
            # Other factors
            if baseline_user['early_login_2024']:
                use_prob += treatment_config['early_login_boost']
                
            if user['user_type'] == 'returning':
                use_prob += treatment_config['returning_user_boost']
                
            # Income effects
            income_effects = treatment_config['income_effects']
            if user['income_bracket'] in ['>150k', '100k-150k']:
                use_prob += income_effects['high_income_boost']
            elif user['income_bracket'] == '<30k':
                use_prob += income_effects['low_income_penalty']  # negative value
            
            # Apply bounds
            use_prob = max(treatment_config['min_probability'], 
                          min(treatment_config['max_probability'], use_prob))
            used_smart_assistant = np.random.random() < use_prob
            
            treatment_data.append({
                'user_id': user['user_id'],
                'used_smart_assistant': used_smart_assistant
            })
            
        return pd.DataFrame(treatment_data)
    
    def generate_2024_outcomes(self, users_df: pd.DataFrame, 
                              baseline_df: pd.DataFrame,
                              treatment_df: pd.DataFrame) -> pd.DataFrame:
        """Generate 2024 outcomes (post-treatment)."""
        
        outcomes_config = self.config['outcomes_2024']
        outcome_data = []
        
        for _, user in users_df.iterrows():
            baseline_user = baseline_df[baseline_df['user_id'] == user['user_id']].iloc[0]
            treatment_user = treatment_df[treatment_df['user_id'] == user['user_id']].iloc[0]
            
            # Main outcome: Filed in 2024 (conversion)
            filing_config = outcomes_config['filing']
            
            if filing_config['apply_demographic_effects']:
                # Use 2023 filing config as base but with 2024 base rate
                base_filing_config = self.config['baseline_2023']['filing'].copy()
                base_filing_config['base_rate'] = filing_config['base_rate']
                base_filing_prob = self._get_filing_probability(user, base_filing_config)
            else:
                base_filing_prob = filing_config['base_rate']
            
            # Treatment effect of Smart Assistant
            treatment_effect = 0
            if treatment_user['used_smart_assistant']:
                treatment_effects = filing_config['treatment_effects']
                base_effect = treatment_effects['base_effect']
                
                # Heterogeneous effects
                if user['tech_savviness'] < 40:
                    base_effect += treatment_effects['low_tech_boost']
                elif user['tech_savviness'] > 70:
                    base_effect += treatment_effects['high_tech_reduction']  # negative value
                    
                if user['age'] > 55:
                    base_effect += treatment_effects['older_user_boost']
                    
                if user['user_type'] == 'new':
                    base_effect += treatment_effects['new_user_boost']
                    
                treatment_effect = base_effect
            
            # Persistence from previous year
            if baseline_user['filed_2023']:
                base_filing_prob += filing_config['filed_2023_boost']
            
            filing_prob_2024 = min(filing_config['max_probability'], 
                                  base_filing_prob + treatment_effect)
            filed_2024 = np.random.random() < filing_prob_2024
            
            # Secondary outcomes
            if filed_2024:
                # Time to complete (minutes)
                time_config = outcomes_config['time_to_complete']
                base_time = baseline_user['time_to_complete_2023'] if baseline_user['time_to_complete_2023'] > 0 else 120
                
                if treatment_user['used_smart_assistant']:
                    time_reduction = np.random.uniform(
                        time_config['treatment_time_reduction']['min_reduction'],
                        time_config['treatment_time_reduction']['max_reduction']
                    )
                    base_time *= (1 - time_reduction)
                
                time_to_complete_2024 = max(time_config['min_time'], 
                                          int(np.random.normal(base_time, time_config['std'])))
                
                # Number of sessions
                session_config = outcomes_config['sessions']
                base_sessions = max(1, baseline_user['sessions_2023'])
                
                if treatment_user['used_smart_assistant']:
                    sessions_reduction = np.random.uniform(
                        session_config['treatment_session_reduction']['min_reduction'],
                        session_config['treatment_session_reduction']['max_reduction']
                    )
                    base_sessions *= (1 - sessions_reduction)
                    
                sessions_2024 = max(1, int(np.random.poisson(base_sessions)))
                
                # Support tickets
                support_config = outcomes_config['support_tickets']
                base_support_rate = support_config['base_rate']
                
                if baseline_user['support_tickets_2023'] > 0:
                    base_support_rate += support_config['filed_2023_penalty']
                    
                if treatment_user['used_smart_assistant']:
                    base_support_rate *= support_config['treatment_reduction']
                    
                support_tickets_2024 = np.random.poisson(base_support_rate)
                
                # User satisfaction (1-10 scale)
                satisfaction_config = outcomes_config['satisfaction']
                base_satisfaction = satisfaction_config['base_score']
                
                if treatment_user['used_smart_assistant']:
                    base_satisfaction += satisfaction_config['treatment_boost']
                if user['tech_savviness'] > 60:
                    base_satisfaction += satisfaction_config['high_tech_boost']
                if baseline_user['support_tickets_2023'] > 0:
                    base_satisfaction += satisfaction_config['support_history_penalty']  # negative value
                    
                satisfaction_2024 = max(satisfaction_config['min_score'], 
                                      min(satisfaction_config['max_score'], 
                                          np.random.normal(base_satisfaction, satisfaction_config['std'])))
                
            else:
                time_to_complete_2024 = 0
                sessions_2024 = 0
                support_tickets_2024 = 0
                satisfaction_2024 = np.nan
            
            outcome_data.append({
                'user_id': user['user_id'],
                'filed_2024': filed_2024,
                'time_to_complete_2024': time_to_complete_2024,
                'sessions_2024': sessions_2024,
                'support_tickets_2024': support_tickets_2024,
                'satisfaction_2024': satisfaction_2024
            })
            
        return pd.DataFrame(outcome_data)
    
    def _get_filing_probability(self, user: pd.Series, filing_config: Dict) -> float:
        """Calculate base filing probability based on user characteristics."""
        prob = filing_config['base_rate']
        
        # Income effects
        income_effects = filing_config['income_effects']
        if user['income_bracket'] in ['75k-100k', '100k-150k', '>150k']:
            prob += income_effects['high_income_boost']
        elif user['income_bracket'] == '<30k':
            prob += income_effects['low_income_penalty']  # negative value
            
        # User type effects
        if user['user_type'] == 'returning':
            prob += filing_config['returning_user_boost']
        
        # Age effects
        if filing_config['prime_age_min'] <= user['age'] <= filing_config['prime_age_max']:
            prob += filing_config['prime_age_boost']
        elif user['age'] > 65 or user['age'] < 25:
            prob += filing_config['non_prime_penalty']  # negative value
            
        return max(0.1, min(0.95, prob))
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate the complete synthetic dataset."""
        print("Generating user demographics...")
        users_df = self.generate_user_demographics()
        
        print("Generating 2023 baseline data...")
        baseline_df = self.generate_2023_baseline_data(users_df)
        
        print("Generating treatment assignment...")
        treatment_df = self.generate_treatment_assignment(users_df, baseline_df)
        
        print("Generating 2024 outcomes...")
        outcomes_df = self.generate_2024_outcomes(users_df, baseline_df, treatment_df)
        
        # Merge all dataframes
        complete_df = users_df.merge(baseline_df, on='user_id')
        complete_df = complete_df.merge(treatment_df, on='user_id')
        complete_df = complete_df.merge(outcomes_df, on='user_id')
        
        # Add derived features if configured
        output_config = self.config['output']
        if output_config['include_derived_features']:
            for feature in output_config['derived_features']:
                if feature == 'time_improvement':
                    complete_df['time_improvement'] = complete_df['time_to_complete_2023'] - complete_df['time_to_complete_2024']
                elif feature == 'session_improvement':
                    complete_df['session_improvement'] = complete_df['sessions_2023'] - complete_df['sessions_2024']
        
        print(f"Generated dataset with {len(complete_df)} users")
        print(f"Treatment rate: {complete_df['used_smart_assistant'].mean():.2%}")
        print(f"2024 filing rate: {complete_df['filed_2024'].mean():.2%}")
        
        return complete_df


def generate_and_save_data(output_path: str = None, 
                          n_users: int = None,
                          config_path: str = "config/simulation_config.yaml") -> pd.DataFrame:
    """
    Generate and save the complete synthetic dataset.
    
    Args:
        output_path: Path to save the CSV file (uses config default if None)
        n_users: Number of users to generate (uses config default if None)
        config_path: Path to the YAML configuration file
        
    Returns:
        Generated DataFrame
    """
    # Load config to get defaults
    config = load_config(config_path)
    
    if output_path is None:
        output_path = config['output']['default_path']
    
    simulator = TaxSoftwareDataSimulator(n_users=n_users, config_path=config_path)
    df = simulator.generate_complete_dataset()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total users: {len(df):,}")
    print(f"Treatment rate: {df['used_smart_assistant'].mean():.1%}")
    print(f"2023 filing rate: {df['filed_2023'].mean():.1%}")
    print(f"2024 filing rate: {df['filed_2024'].mean():.1%}")
    
    # Treatment vs control outcomes
    treated = df[df['used_smart_assistant'] == True]
    control = df[df['used_smart_assistant'] == False]
    
    print(f"\nTreated group filing rate: {treated['filed_2024'].mean():.1%}")
    print(f"Control group filing rate: {control['filed_2024'].mean():.1%}")
    print(f"Naive treatment effect: {(treated['filed_2024'].mean() - control['filed_2024'].mean()):.1%}")
    
    return df


if __name__ == "__main__":
    # Generate the dataset using configuration
    df = generate_and_save_data() 