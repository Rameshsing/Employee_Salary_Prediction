import pandas as pd
import numpy as np
import joblib

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    
    # Replace '?' with 'Unknown'
    df.replace('?', 'Unknown', inplace=True)
    
    # Strip whitespace
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    return df

def get_feature_info():
    """Return information about dataset features"""
    feature_info = {
        'age': 'Age of the individual',
        'workclass': 'Type of employment (Private, Government, etc.)',
        'fnlwgt': 'Final weight - demographic sampling weight',
        'educational-num': 'Education level in numeric form (higher = more education)',
        'marital-status': 'Marital status of the individual',
        'occupation': 'Type of job/occupation',
        'relationship': 'Family relationship role',
        'race': 'Race/ethnicity',
        'gender': 'Gender (Male/Female)',
        'capital-gain': 'Capital gains from investments',
        'capital-loss': 'Capital losses from investments',
        'hours-per-week': 'Number of hours worked per week',
        'native-country': 'Country of birth',
        'income': 'Target variable: <=50K or >50K'
    }
    return feature_info

def validate_input(input_dict):
    """Validate user input data"""
    errors = []
    
    if input_dict['age'] < 16 or input_dict['age'] > 100:
        errors.append("Age must be between 16 and 100")
    
    if input_dict['hours-per-week'] < 1 or input_dict['hours-per-week'] > 168:
        errors.append("Hours per week must be between 1 and 168")
    
    if input_dict['educational-num'] < 1 or input_dict['educational-num'] > 16:
        errors.append("Education number must be between 1 and 16")
    
    return errors

def generate_sample_data():
    """Generate sample employee data for testing"""
    samples = [
        {
            'name': 'High Earner Profile',
            'data': {
                'age': 45,
                'workclass': 'Private',
                'fnlwgt': 200000,
                'educational-num': 16,
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Husband',
                'race': 'White',
                'gender': 'Male',
                'capital-gain': 5000,
                'capital-loss': 0,
                'hours-per-week': 50,
                'native-country': 'United-States'
            }
        },
        {
            'name': 'Lower Earner Profile',
            'data': {
                'age': 25,
                'workclass': 'Private',
                'fnlwgt': 100000,
                'educational-num': 9,
                'marital-status': 'Never-married',
                'occupation': 'Handlers-cleaners',
                'relationship': 'Not-in-family',
                'race': 'White',
                'gender': 'Female',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 30,
                'native-country': 'United-States'
            }
        }
    ]
    return samples
