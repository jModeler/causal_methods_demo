"""
Data simulation module for generating realistic tax software user data.

This module creates synthetic data for demonstrating causal inference methods
in the context of analyzing the impact of a "Smart Filing Assistant" feature
on user conversion and engagement.
"""

import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
fake.seed_instance(42)


class TaxSoftwareDataSimulator:
    """Simulates realistic tax software user data with causal relationships."""
    
    def __init__(self, n_users: int = 10000):
        """
        Initialize the data simulator.
        
        Args:
            n_users: Number of users to generate
        """
        self.n_users = n_users
        self.fake = fake
        
    def generate_user_demographics(self) -> pd.DataFrame:
        """Generate user demographic and metadata features."""
        
        # Income brackets (affects both feature adoption and conversion)
        income_brackets = ['<30k', '30k-50k', '50k-75k', '75k-100k', '100k-150k', '>150k']
        income_weights = [0.15, 0.20, 0.25, 0.20, 0.15, 0.05]
        
        # Device types
        device_types = ['mobile', 'desktop', 'tablet']
        device_weights = [0.45, 0.50, 0.05]
        
        # User types
        user_types = ['new', 'returning']
        user_type_weights = [0.35, 0.65]
        
        # Geographic regions (affects tech adoption)
        regions = ['West', 'East', 'Midwest', 'South']
        region_weights = [0.25, 0.30, 0.20, 0.25]
        
        data = []
        for i in range(self.n_users):
            user_id = f"user_{i:06d}"
            income_bracket = np.random.choice(income_brackets, p=income_weights)
            device_type = np.random.choice(device_types, p=device_weights)
            user_type = np.random.choice(user_types, p=user_type_weights)
            region = np.random.choice(regions, p=region_weights)
            
            # Age correlated with tech-savviness
            age = int(np.random.normal(45, 15))
            age = max(18, min(80, age))  # Clip to reasonable range
            
            # Tech-savviness score (hidden confounders)
            # Younger users, higher income, West coast tend to be more tech-savvy
            tech_base = 50
            if age < 35:
                tech_base += 20
            elif age > 55:
                tech_base -= 15
                
            if income_bracket in ['>150k', '100k-150k']:
                tech_base += 15
            elif income_bracket == '<30k':
                tech_base -= 10
                
            if region == 'West':
                tech_base += 10
            elif region in ['Midwest', 'South']:
                tech_base -= 5
                
            tech_savviness = max(0, min(100, int(np.random.normal(tech_base, 15))))
            
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
        
        baseline_data = []
        
        for _, user in users_df.iterrows():
            # Filing behavior in 2023
            filed_2023 = np.random.random() < self._get_filing_probability(user, base_rate=0.75)
            
            if filed_2023:
                # Time to complete filing (minutes)
                base_time = 120
                if user['user_type'] == 'returning':
                    base_time -= 30
                if user['tech_savviness'] > 70:
                    base_time -= 20
                elif user['tech_savviness'] < 30:
                    base_time += 40
                    
                time_to_complete_2023 = max(30, int(np.random.normal(base_time, 30)))
                
                # Number of sessions
                base_sessions = 2.5
                if user['device_type'] == 'mobile':
                    base_sessions += 0.5
                if user['tech_savviness'] < 40:
                    base_sessions += 1
                    
                sessions_2023 = max(1, int(np.random.poisson(base_sessions)))
                
                # Support tickets
                support_prob = 0.15
                if user['tech_savviness'] < 30:
                    support_prob += 0.15
                if user['age'] > 65:
                    support_prob += 0.10
                    
                support_tickets_2023 = np.random.poisson(support_prob)
            else:
                time_to_complete_2023 = 0
                sessions_2023 = 0
                support_tickets_2023 = 0
            
            # Early login behavior (predictor of feature adoption)
            early_login_prob = 0.3
            if user['tech_savviness'] > 60:
                early_login_prob += 0.2
            if user['user_type'] == 'returning':
                early_login_prob += 0.15
                
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
        
        treatment_data = []
        
        for _, user in users_df.iterrows():
            baseline_user = baseline_df[baseline_df['user_id'] == user['user_id']].iloc[0]
            
            # Probability of using Smart Filing Assistant
            # This is the key selection bias - tech-savvy users more likely to adopt
            use_prob = 0.4  # Base probability
            
            # Tech-savviness is main driver
            if user['tech_savviness'] > 70:
                use_prob += 0.25
            elif user['tech_savviness'] > 50:
                use_prob += 0.10
            elif user['tech_savviness'] < 30:
                use_prob -= 0.20
            
            # Other factors
            if user['age'] < 35:
                use_prob += 0.15
            elif user['age'] > 55:
                use_prob -= 0.10
                
            if user['device_type'] == 'mobile':
                use_prob += 0.05
            elif user['device_type'] == 'tablet':
                use_prob -= 0.05
                
            if baseline_user['early_login_2024']:
                use_prob += 0.20
                
            if user['user_type'] == 'returning':
                use_prob += 0.05
                
            # Higher income users more willing to try new features
            if user['income_bracket'] in ['>150k', '100k-150k']:
                use_prob += 0.10
            elif user['income_bracket'] == '<30k':
                use_prob -= 0.05
            
            use_prob = max(0.05, min(0.95, use_prob))
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
        
        outcome_data = []
        
        for _, user in users_df.iterrows():
            baseline_user = baseline_df[baseline_df['user_id'] == user['user_id']].iloc[0]
            treatment_user = treatment_df[treatment_df['user_id'] == user['user_id']].iloc[0]
            
            # Main outcome: Filed in 2024 (conversion)
            base_filing_prob = self._get_filing_probability(user, base_rate=0.72)
            
            # Treatment effect of Smart Assistant
            treatment_effect = 0
            if treatment_user['used_smart_assistant']:
                # Positive treatment effect, but varies by user characteristics
                base_effect = 0.08  # 8 percentage point increase
                
                # Heterogeneous effects
                if user['tech_savviness'] < 40:
                    # Low tech users benefit more from assistance
                    base_effect += 0.05
                elif user['tech_savviness'] > 70:
                    # High tech users benefit less (already efficient)
                    base_effect -= 0.02
                    
                if user['age'] > 55:
                    # Older users benefit more
                    base_effect += 0.03
                    
                if user['user_type'] == 'new':
                    # New users benefit more from guidance
                    base_effect += 0.04
                    
                treatment_effect = base_effect
            
            # Strong persistence from previous year
            if baseline_user['filed_2023']:
                base_filing_prob += 0.15
            
            filing_prob_2024 = min(0.95, base_filing_prob + treatment_effect)
            filed_2024 = np.random.random() < filing_prob_2024
            
            # Secondary outcomes
            if filed_2024:
                # Time to complete (minutes) - Smart Assistant should reduce time
                base_time = baseline_user['time_to_complete_2023'] if baseline_user['time_to_complete_2023'] > 0 else 120
                
                if treatment_user['used_smart_assistant']:
                    # Smart Assistant reduces time by 15-25%
                    time_reduction = np.random.uniform(0.15, 0.25)
                    base_time *= (1 - time_reduction)
                
                time_to_complete_2024 = max(20, int(np.random.normal(base_time, 20)))
                
                # Number of sessions - Smart Assistant should reduce sessions
                base_sessions = max(1, baseline_user['sessions_2023'])
                if treatment_user['used_smart_assistant']:
                    sessions_reduction = np.random.uniform(0.10, 0.30)
                    base_sessions *= (1 - sessions_reduction)
                    
                sessions_2024 = max(1, int(np.random.poisson(base_sessions)))
                
                # Support tickets - Smart Assistant should reduce support needs
                base_support_rate = 0.12
                if baseline_user['support_tickets_2023'] > 0:
                    base_support_rate += 0.05
                    
                if treatment_user['used_smart_assistant']:
                    base_support_rate *= 0.6  # 40% reduction
                    
                support_tickets_2024 = np.random.poisson(base_support_rate)
                
                # User satisfaction (1-10 scale)
                base_satisfaction = 7.2
                if treatment_user['used_smart_assistant']:
                    base_satisfaction += 0.8
                if user['tech_savviness'] > 60:
                    base_satisfaction += 0.3
                if baseline_user['support_tickets_2023'] > 0:
                    base_satisfaction -= 0.5
                    
                satisfaction_2024 = max(1, min(10, np.random.normal(base_satisfaction, 1.2)))
                
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
    
    def _get_filing_probability(self, user: pd.Series, base_rate: float = 0.75) -> float:
        """Calculate base filing probability based on user characteristics."""
        prob = base_rate
        
        # Income effect
        if user['income_bracket'] in ['>150k', '100k-150k', '75k-100k']:
            prob += 0.10
        elif user['income_bracket'] == '<30k':
            prob -= 0.15
            
        # Returning users more likely to file
        if user['user_type'] == 'returning':
            prob += 0.08
        
        # Age effect (middle-aged most likely to file)
        if 25 <= user['age'] <= 55:
            prob += 0.05
        elif user['age'] > 65 or user['age'] < 25:
            prob -= 0.05
            
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
        
        # Add some additional derived features
        complete_df['time_improvement'] = complete_df['time_to_complete_2023'] - complete_df['time_to_complete_2024']
        complete_df['session_improvement'] = complete_df['sessions_2023'] - complete_df['sessions_2024']
        
        print(f"Generated dataset with {len(complete_df)} users")
        print(f"Treatment rate: {complete_df['used_smart_assistant'].mean():.2%}")
        print(f"2024 filing rate: {complete_df['filed_2024'].mean():.2%}")
        
        return complete_df


def generate_and_save_data(output_path: str = "data/simulated_users.csv", 
                          n_users: int = 10000) -> pd.DataFrame:
    """
    Generate and save the complete synthetic dataset.
    
    Args:
        output_path: Path to save the CSV file
        n_users: Number of users to generate
        
    Returns:
        Generated DataFrame
    """
    simulator = TaxSoftwareDataSimulator(n_users=n_users)
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
    # Generate the dataset
    df = generate_and_save_data() 