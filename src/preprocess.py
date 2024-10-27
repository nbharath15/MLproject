import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    return pd.read_csv(filepath)

def scale_features(X):
    """Scales continuous features (e.g., distances, elevations)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess_data(data):
    """Main function for preprocessing the data."""
    # Ensure 'Cover_Type' exists in the dataset
    if 'Cover_Type' not in data.columns:
        raise ValueError("'Cover_Type' column is missing from the dataset.")

    # Separate features and target variable
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']
    
    # Identify continuous features to scale
    continuous_features = [
        'Elevation', 
        'Slope', 
        'Horizontal_Distance_To_Hydrology', 
        'Vertical_Distance_To_Hydrology', 
        'Horizontal_Distance_To_Roadways'
    ]
    
    # Ensure continuous features exist in the dataset
    missing_features = [feature for feature in continuous_features if feature not in X.columns]
    if missing_features:
        raise ValueError(f"Missing continuous features: {missing_features}")

    # Scale continuous features
    X_scaled = scale_features(X[continuous_features])
    
    # Replace original continuous features with scaled features
    X[continuous_features] = X_scaled

    return X, y
