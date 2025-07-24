# File: src/modeling/train.py (Refactored for Daily Features)
"""
This script trains a LightGBM model for each target city using the
feature-rich DAILY dataset created by the build_daily_features.py script.
"""

import pandas as pd
import lightgbm as lgb
import os
import sys
import joblib

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Configuration ---
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files", "Master_Daily_Features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TARGET_VARIABLE = "AQI"


def train_and_save_models(data_path: str, models_dir: str):
    """
    Loads the daily feature dataset, trains a LightGBM model for each city, and saves them.
    """
    print("--- Starting Model Training on Daily Data ---")
    
    try:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        print(f"Successfully loaded data. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Daily feature data file not found at '{data_path}'.")
        print("Please run scripts/build_daily_features.py first.")
        return

    os.makedirs(models_dir, exist_ok=True)
    
    cities = df['City'].unique()
    print(f"\nFound cities in dataset: {cities}")
    
    for city in cities:
        print(f"\n--- Training Model for: {city} ---")
        
        city_df = df[df['City'] == city].copy()
        
        # Define features (X) and target (y)
        # We drop 'Date' and 'City' as they are not features for the model.
        features = [col for col in city_df.columns if col not in ['Date', 'City', TARGET_VARIABLE]]
        X = city_df[features]
        y = city_df[TARGET_VARIABLE]
        
        print(f"Training with {len(X)} samples and {len(features)} features.")
        
        # Define the LightGBM model with final, robust Hyperparameter tuning
        lgbm = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=300,         # A moderate number of trees
            learning_rate=0.05,       # A standard learning rate
            num_leaves=20,            # Simple trees
            max_depth=7,              # Even shallower trees to force generalization
            
            # --- These are the key new parameters for stability ---
            subsample=0.7,            # Use only 70% of data for each tree
            subsample_freq=1,         # Perform bagging at every iteration
            colsample_bytree=0.7,     # Use only 70% of features for each tree
            
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=0.1,           # L2 regularization
            random_state=42,
            n_jobs=-1
        )
        
        print("Starting training...")
        lgbm.fit(X, y)
        print("Training complete.")
        
        model_filename = f"{city}_lgbm_daily_model.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        try:
            joblib.dump(lgbm, model_path)
            print(f"Successfully saved model for {city} to: {model_path}")
        except Exception as e:
            print(f"ERROR: Could not save model for {city}. Reason: {e}")

    print("\n--- All Models Trained Successfully ---")

# --- Main Execution ---
if __name__ == "__main__":
    train_and_save_models(DATA_PATH, MODELS_DIR)