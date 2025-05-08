
import torch
import time
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


sys.path.append('/home/../multiTS/NFT/models/baseline_models')
from base_models import TCN


sys.path.append('/home/../multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_num_of_series, data_to_steps, single_data_to_series_list
from models.training_functions import train_model, evaluate_model

sys.path.append('NFT/')
from models.training_functions import get_data, evaluate_model, add_results_to_excel, get_path


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataset(Dataset):
    def __init__(self, data, target):
        self.input = data.to(device)
        self.target = target.to(device)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx): 
        return self.input[idx], self.target[idx]

def main(
        data='noaa',
        lookback=15,
        horizon=1,
        num_epochs=10,
        plot_epoch=100,
        batch_size=32,
        num_channels=[25,50],
        kernel_size=2,
        dropout=0.2, 
        series=None,
):
    
    num_of_vars=data_to_num_vars_dict[data]    
    n_series=data_to_num_of_series[data]
            
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
        data=data, 
        lookback=lookback, 
        horizon=horizon,
        n_series=n_series,
        print_stats=False,
        series=series
        )

    model = TCN(
            lookback=lookback, 
            horizon=horizon, 
            num_vars=num_of_vars, 
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            dropout=dropout
            ).to(device)
   
 
    train_model(
        dataset=dataset,
        epochs=num_epochs,
        model=model, 
        train_X=train_X.to(device), train_y=train_y.to(device),
        val_X=val_X.to(device), val_y=val_y.to(device), 
        device=device, 
        lookback=lookback,
        horizon=horizon, 
        n_vars=num_of_vars, 
        batch_size=batch_size, 
        print_epoch=10000000, 
        path_to_save_prediction_plots=None
        )
    
    _, _, _, train_mse, test_mse, train_mae, test_mae, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase = evaluate_model(
        model, 
        train_X.to(device), 
        train_y.to(device), 
        val_X.to(device), 
        val_y.to(device), 
        test_X.to(device), 
        test_y.to(device)
        )

    add_results_to_excel('tcn', data, lookback, horizon, num_epochs, None, None, 
                         fourier_granularity=None, poly_degree=None, 
                         num_channels=num_channels, series=series, train_mse=train_mse, test_mse=test_mse, 
                         train_mae=train_mae, test_mae=test_mae, train_smape=train_smape, test_smape=test_smape, train_mape=train_mape,
                         test_mape=test_mape, train_mase=train_mase, test_mase=test_mase, train_rmsse=None, test_rmsse=None)
    


if __name__ == "__main__":
    data = 'chorales'
    print(f'data={data}')
    
    for lookback, horizon in data_to_steps[data]:
        if data in ['eeg_single', 'ecg_single', 'noaa', 'mini_electricity', 'air_quality_seasonal', 'air_quality_seasonal_2_var']:
            for series in single_data_to_series_list[data]:
              main(
                data=data,
                lookback=lookback,
                horizon=horizon,
                num_epochs=10,
                plot_epoch=100,
                batch_size=32,
                series=series,#'E00001', # "AEM00041194" #"AG000060590"
                )              
        else:            
            main(
                data=data,
                lookback=lookback,
                horizon=horizon,
                num_epochs=10,
                plot_epoch=100,
                batch_size=32,
                series=None,#'E00001', # "AEM00041194" #"AG000060590"
                )