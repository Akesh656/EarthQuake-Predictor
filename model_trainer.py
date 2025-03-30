import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_processor import load_and_clean_data, prepare_features, create_preprocessing_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(features, target, test_size=0.2, random_state=42):
    """
    Train a machine learning model to predict earthquake magnitude
    """
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        
        # Scale numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[['Latitude', 'Longitude', 'Depth']])
        X_test_scaled = scaler.transform(X_test[['Latitude', 'Longitude', 'Depth']])
        
        # Initialize models
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        
        # Train and evaluate models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{name} - MSE: {mse:.4f}, R^2: {r2:.4f}")
            
            # Keep track of the best model
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name
        
        logger.info(f"Best model: {best_model_name} with R^2: {best_score:.4f}")
        
        # Save the best model and scaler
        with open('earthquake_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info("Model and scaler saved successfully.")
        
        return best_model, scaler, best_score
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def main():
    """
    Main function to load data, train model, and save results
    """
    try:
        logger.info("Starting model training process...")
        
        # Load and preprocess data
        filepath = 'attached_assets/Earthquake_Data (1).csv'
        if not os.path.exists(filepath):
            logger.error(f"Data file not found at {filepath}")
            return
            
        df = load_and_clean_data(filepath)
        
        # Use a smaller sample of data (10%) for faster processing
        sample_size = min(2000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        logger.info(f"Using a sample of {sample_size} records for training")
        
        features, target = prepare_features(df_sample)
        
        # Train model
        model, scaler, score = train_model(features, target)
        
        logger.info("Model training completed.")
        return model, scaler, score
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
