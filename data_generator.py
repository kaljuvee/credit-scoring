import pandas as pd
import numpy as np

def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    data = {
        'income': np.random.normal(50000, 20000, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(200000, 100000, n_samples),
    }
    
    # Generate categorical features
    locations = ['Urban', 'Suburban', 'Rural']
    education = ['High School', 'Bachelor', 'Master', 'PhD']
    
    data['location'] = np.random.choice(locations, n_samples)
    data['education'] = np.random.choice(education, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (default_rate)
    # Higher probability of default if:
    # - income is low
    # - credit score is low
    # - loan amount is high relative to income
    prob_default = 1 / (1 + np.exp(-(
        -5 
        - df['income']/50000 
        + df['loan_amount']/100000 
        - df['credit_score']/200
        + (df['age'] < 25).astype(int)
    )))
    
    df['default'] = np.random.binomial(1, prob_default)
    
    return df 