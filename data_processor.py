import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
filepath = r"C:\Users\IKSHITHA\Desktop\EarthquakePredictorML\Earthquake_Data (1).csv"
def load_and_clean_data(filepath):
    """
    Load and preprocess earthquake data from CSV file
    """
    try:
        # First, read the header to get column names
        with open(filepath, 'r') as f:
            header = f.readline().strip()
        
        # Load the dataset using whitespace separator, skip header
        df = pd.read_csv(filepath, skiprows=1, sep='\s+', 
                         names=['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 
                               'Magnitude', 'MagType', 'NST', 'Gap', 'Clo', 
                               'RMS', 'SRC', 'EventID'])
        
        # Convert types
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
        df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
        
        # Drop rows with missing values in key columns
        df = df.dropna(subset=['Latitude', 'Longitude', 'Depth', 'Magnitude'])
        
        # Extract year and month from date
        df['Year'] = df['Date'].str.split('/').str[0].astype(int)
        df['Month'] = df['Date'].str.split('/').str[1].astype(int)
        
        # Create some new features
        df['DepthCategory'] = pd.cut(df['Depth'], 
                                     bins=[0, 10, 30, 100, 700], 
                                     labels=['Shallow', 'Intermediate', 'Deep', 'Very Deep'])
        
        # Classify earthquakes by magnitude
        df['MagnitudeCategory'] = pd.cut(df['Magnitude'], 
                                         bins=[0, 3, 5, 7, 9, 10], 
                                         labels=['Minor', 'Light', 'Moderate', 'Major', 'Great'])
        
        # Create a location feature (simplified for demonstration)
        conditions = [
            (df['Latitude'] > 30) & (df['Latitude'] < 42) & (df['Longitude'] > -125) & (df['Longitude'] < -115),
            (df['Latitude'] > 30) & (df['Latitude'] < 45) & (df['Longitude'] > 125) & (df['Longitude'] < 145),
            (df['Latitude'] > -10) & (df['Latitude'] < 10) & (df['Longitude'] > 95) & (df['Longitude'] < 130),
            (df['Latitude'] > -40) & (df['Latitude'] < -15) & (df['Longitude'] > -75) & (df['Longitude'] < -65)
        ]
        choices = ['California', 'Japan', 'Indonesia', 'Chile']
        df['Location'] = np.select(conditions, choices, default='Other')
        
        logger.info(f"Data loaded and cleaned. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading or cleaning data: {e}")
        raise

def prepare_features(df):
    """
    Prepare features for machine learning model
    """
    try:
        # Select features
        features = df[['Latitude', 'Longitude', 'Depth', 'Year', 'Month', 'Location']]
        target = df['Magnitude']
        
        logger.info(f"Features prepared. Features shape: {features.shape}, Target shape: {target.shape}")
        return features, target
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for feature transformation
    """
    # Numeric features
    numeric_features = ['Latitude', 'Longitude', 'Depth', 'Year', 'Month']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features
    categorical_features = ['Location']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def preprocess_input(latitude, longitude, depth, scaler=None):
    """
    Preprocess a single input for prediction
    """
    try:
        # Create a feature vector with the same format as training data
        # We're making a simplified version that only includes the numeric features
        # that are compatible with our model
        features = np.array([[latitude, longitude, depth]])
        
        # Scale features if scaler is provided
        if scaler:
            features = scaler.transform(features)
            
        return features
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        # If an error occurs, return a default feature vector
        return np.array([[latitude, longitude, depth]])

if __name__ == "__main__":
    # Test data loading and preprocessing
    try:
        df = load_and_clean_data('attached_assets/Earthquake_Data (1).csv')
        features, target = prepare_features(df)
        print(f"Data loaded successfully. Sample features:\n{features.head()}")
        print(f"Sample target:\n{target.head()}")
    except Exception as e:
        print(f"Error in data processing: {e}")
