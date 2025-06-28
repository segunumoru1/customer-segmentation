from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline for the dataset."""
    features = ['annual_income', 'spending_score', 'savings', 'age']
    preprocessor = StandardScaler()
    return preprocessor, features

def preprocess_data(df, preprocessor, features):
    """Preprocess the data using the provided preprocessor and features."""
    preprocessor, features = create_preprocessing_pipeline()
    df[features] = preprocessor.fit_transform(df[features])
    return df, preprocessor, features