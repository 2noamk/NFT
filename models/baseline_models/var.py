import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import sys
sys.path.append('/home/../multiTS/NFT/models/NFT')
from NFT import NFT

import torch.nn.functional as F

sys.path.append('NFT/')
from dicts import data_to_num_vars_dict, data_to_label_len, data_to_num_nft_blocks, data_to_num_of_series, data_to_steps, single_data_to_series_list, noaa_series_to_years
from lists import ECG

sys.path.append('NFT/')
from models.training_functions import get_data, evaluate_model, save_model, get_model_name, add_results_to_excel, get_path


def reshape_test_y(test_y):
    """
    Reshape test_y (3D) into 2D to match forecasted values.

    Args:
        test_y (np.array or pd.DataFrame): Test data with shape (num_data_points, horizon, num_vars).

    Returns:
        pd.DataFrame: Reshaped data with shape (num_data_points * horizon, num_vars).
    """
    # Combine all samples into one horizon set
    reshaped = test_y.reshape(-1, test_y.shape[-1])  # Combine all rows
    return pd.DataFrame(reshaped)

def reshape_data_for_var(data):
    num_data_points, lookback, num_vars = data.shape
    reshaped = data.reshape(num_data_points * lookback, num_vars)
    return pd.DataFrame(reshaped)

def forecast_with_var(train, horizon):
    # Train the VAR model
    model = VAR(train)
    fitted_model = model.fit()

    # Forecast for the given horizon
    lag_order = fitted_model.k_ar
    forecast_input = train.values[-lag_order:]  # Use last 'lag_order' observations
    forecast = fitted_model.forecast(y=forecast_input, steps=horizon)

    # Convert forecast to DataFrame
    forecast_df = pd.DataFrame(forecast, columns=train.columns)
    return forecast_df

def calculate_mse(forecast_df, test):
    """
    Calculate Mean Squared Error (MSE) between forecasted and actual values.

    Args:
        forecast_df (pd.DataFrame): Forecasted values with shape (horizon, num_vars).
        test (pd.DataFrame): Actual values with shape (horizon, num_vars).

    Returns:
        float: MSE value.
    """
    # Ensure the number of samples matches
    if len(forecast_df) > len(test):
        forecast_df = forecast_df.iloc[:len(test)]
    elif len(forecast_df) < len(test):
        test = test.iloc[:len(forecast_df)]
    
    mse = mean_squared_error(test.values.flatten(), forecast_df.values.flatten())
    mae = mean_absolute_error(test.values.flatten(), forecast_df.values.flatten())
    # mse = F.mse_loss(test.values, forecast_df.values)
    return mse, mae

def calculate_var_mse(data, lookback, horizon, series):          
    num_of_vars=data_to_num_vars_dict.get(data, 5) 
    n_series=data_to_num_of_series.get(data, 1)
   
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
        data=data, 
        lookback=lookback, 
        horizon=horizon,
        n_series=n_series,
        print_stats=False,
        series=series,
        )
    

    # Reshape train_X and test_X for VAR
    train_data = reshape_data_for_var(train_X)

    # Forecast
    horizon = test_y.shape[1]  # Horizon length
    forecasted_values = forecast_with_var(train_data, horizon)

    # Reshape test_y properly
    test_y_reshaped = reshape_test_y(test_y)

    # Calculate MSE
    mse, mae = calculate_mse(forecasted_values, test_y_reshaped)
    
    return mse, mae
    
    
for data in ['noaa']:
    print(data)
    for lookback, horizon in data_to_steps[data]:
        if data in ['eeg_single', 'ecg_single', 'noaa']:
            mse_lst, mae_lst = [], []
            for series in single_data_to_series_list[data]:
                mse, mae = calculate_var_mse(data, lookback, horizon, series)
                mse_lst.append(mse)
                mae_lst.append(mae)
            mse = sum(mse_lst) / len(mse_lst)
            mae = sum(mae_lst) / len(mae_lst)
        else:
            mse, mae = calculate_var_mse(data, lookback, horizon, series=None)

        # Print results
        print(f"horizon={horizon} (MSE, MAE):", mse, mae)
