# File: tests/modeling/test_train.py

"""
Integration tests for the model training script `src/modeling/train.py`.
"""
import pytest
import pandas as pd
import os
import sys

# --- Setup Project Root Path ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..', '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

from src.modeling.train import train_and_save_models

# --- Helper Fixture for Creating Fake Data ---

@pytest.fixture
def create_fake_feature_dataset(tmp_path):
    """
    Creates a small, temporary feature-rich dataset for testing the training script.
    `tmp_path` is a special pytest fixture that creates a temporary directory.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    date_range = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    num_rows = len(date_range)
    
    data = {
        'Date': date_range,
        'City': ['TestCity'] * num_rows,
        'AQI': [50, 55, 60, 65, 70, 75, 80],
        'latitude': [10.0] * num_rows,
        'longitude': [10.0] * num_rows,
        'day_of_week': [5, 6, 0, 6, 0, 1, 2], 
        'month': [1] * num_rows,
        'year': [2022, 2022, 2022, 2023, 2023, 2023, 2023],
        'AQI_lag_1_day': [45, 50, 55, 60, 65, 70, 75],
        'AQI_lag_7_day': [30, 35, 40, 45, 50, 55, 60],
        'temperature_2m_mean': [20, 21, 22, 23, 24, 25, 26],
        'temperature_2m_min': [15, 16, 17, 18, 19, 20, 21],
        'temperature_2m_max': [25, 26, 27, 28, 29, 30, 31],
        'relative_humidity_2m_mean': [60, 62, 61, 63, 64, 65, 66],
        'precipitation_sum': [0, 0, 1, 0, 0, 2, 0],
        'wind_speed_10m_mean': [5, 6, 7, 4, 5, 6, 7],
        'AQI_rolling_mean_3_day': [40, 45, 50, 55, 60, 65, 70], 
        'AQI_rolling_mean_7_day': [35, 40, 45, 50, 55, 60, 65], 
        'temp_x_humidity': [1200, 1302, 1342, 1449, 1536, 1625, 1716] 
    }
    df = pd.DataFrame(data)
    
    fake_data_path = data_dir / "fake_master_features.csv"
    df.to_csv(fake_data_path, index=False)
    
    return str(fake_data_path)


# --- The Main Test Function ---

def test_train_and_save_models_creates_output_files(tmp_path, create_fake_feature_dataset):
    """
    Tests that the train_and_save_models function runs without errors and
    creates the expected model and metadata files.
    """
    fake_data_path = create_fake_feature_dataset
    
    models_dir = tmp_path / "models"
    
    # --- Execute the function we are testing ---
    train_and_save_models(data_path=fake_data_path, models_dir=str(models_dir))
    
    # --- Assert that the output files were created ---
    expected_city = "TestCity"
    2
    # 1. Check for the model file (.pkl)
    expected_model_file = models_dir / f"{expected_city}_lgbm_daily_model.pkl"
    assert expected_model_file.exists(), "The .pkl model file was not created."
    
    # 2. Check for the feature importance file (.json)
    expected_importance_file = models_dir / f"{expected_city}_feature_importance.json"
    assert expected_importance_file.exists(), "The feature importance JSON was not created."
    
    # 3. Check for the validation score file (.json)
    expected_score_file = models_dir / f"{expected_city}_validation_score.json"
    assert expected_score_file.exists(), "The validation score JSON was not created."