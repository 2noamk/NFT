
import torch
import time
import sys
import numpy as np
import pandas as pd

# sys.path.append('NFT/models/NFT/')
# from NFT_NFT import NFT

sys.path.append('/home/noam.koren/multiTS/NFT/models/NFT')
from NFT import NFT


sys.path.append('NFT/')
from dicts import data_to_num_vars_dict, data_to_num_nft_blocks, data_to_num_of_series, data_to_steps, single_data_to_series_list, noaa_series_to_years
from lists import ECG

sys.path.append('NFT/')
from models.training_functions import get_data, evaluate_model, save_model, get_model_name, add_results_to_excel, get_path


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_forecast_and_thetas(model, test_X, path_to_save_model): 

    forecast, trend_forecast, seasonality_forecast, trend_thetas, seasonality_thetas = model.predict(test_X, return_interpretable=True, return_thetas=True)

    sample_forecast, sample_trend_forecast, sample_seasonality_forecast = forecast[0].numpy(), trend_forecast[0].numpy(), seasonality_forecast[0].numpy()
    trend_thetas, seasonality_thetas = trend_thetas[0].numpy(), seasonality_thetas[0].numpy()
    
    print(sample_forecast.shape, sample_trend_forecast.shape, sample_seasonality_forecast.shape)
    print(trend_thetas.shape, seasonality_thetas.shape)

    # Concatenate the arrays along the second dimension (axis=1)
    d = np.concatenate((sample_forecast, sample_trend_forecast, sample_seasonality_forecast), axis=1)
    thetas = np.concatenate((trend_thetas, seasonality_thetas), axis=1)
    print(thetas.shape)
    # Create column names
    num_series = sample_forecast.shape[1]
    columns = [f'forecast series {i+1}' for i in range(num_series)] + \
            [f'trend series {i+1}' for i in range(num_series)] + \
            [f'seasonality series {i+1}' for i in range(num_series)]
    # Create column names
    num_trend_thetas = trend_thetas.shape[1]
    num_seasonality_thetas = seasonality_thetas.shape[1]
    columns_thetas = [f'trend thetas series {i+1}' for i in range(num_trend_thetas)] + \
            [f'seasonality_thetas series {i+1}' for i in range(num_seasonality_thetas)]
    print(columns_thetas)
    # Create a DataFrame
    df = pd.DataFrame(d, columns=columns)
    # Create a DataFrame
    df_thetas = pd.DataFrame(thetas, columns=columns_thetas)

    # Save to Excel
    df.to_excel(f'{path_to_save_model}/interpret_Results_{data}.xlsx', index=False)
    # Save to Excel
    df_thetas.to_excel(f'{path_to_save_model}/thetas_{data}.xlsx', index=False)

    print(f"Data saved to {path_to_save_model}")

def main(
        data='noaa',
        lookback=15,
        horizon=1,
        num_epochs=10,
        plot_epoch=100,
        blocks=2,
        layers_type='tcn',
        batch_size=32,
        series=None,
        year=None,
        thetas_dim=(4, 8),
        stack_types=('trend', 'seasonality'),
        get_forecast_and_coeffs=False,
        dft_1d=False
):
    
    num_of_vars=data_to_num_vars_dict.get(data, 5) 
    n_series=data_to_num_of_series.get(data, 1)
    
    model_name = get_model_name('nft', data, lookback, horizon, num_epochs, blocks, layers_type, n_series, series)
    
    path_to_save_model, path_to_save_checkpoint_models, path_to_save_loss_plots, path_to_save_prediction_plots = get_path(model_name, data, 'nft')
    
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
        data=data, 
        lookback=lookback, 
        horizon=horizon,
        n_series=n_series,
        print_stats=False,
        series=series,
        year=year,
        )

    model = NFT(
        forecast_length=horizon,
        backcast_length=lookback,
        n_vars=num_of_vars,
        nb_blocks_per_stack=blocks,
        layers_type=layers_type,
        num_channels_for_tcn=[25, 50],
        thetas_dim=thetas_dim,
        stack_types=stack_types,
        )
   
    model.compile(loss='mse', optimizer='adam')
 
    model.fit(
        x_train=train_X, 
        y_train=train_y, 
        validation_data=(val_X, val_y), 
        epochs=num_epochs,
        batch_size=batch_size, 
        plot_epoch=plot_epoch,
        path_to_save_model=path_to_save_checkpoint_models,
        path_to_save_loss_plots=path_to_save_loss_plots,
        path_to_save_prediction_plots=path_to_save_prediction_plots,
        dft_1d=dft_1d
        )
    
    if get_forecast_and_coeffs: get_forecast_and_thetas(model, test_X, path_to_save_model)
    
    
    # start_time = time.time()
    # test_pred = model.predict(test_X).clone().detach()
    # end_time = time.time()
    # prediction_time = end_time - start_time
    # print(f"Prediction time: {prediction_time:.4f} seconds")

    
    train_pred, val_pred, test_pred, train_mse, test_mse, train_mae, test_mae, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase = evaluate_model(model, train_X, train_y, val_X, val_y, test_X, test_y)
    
    # add_results_to_excel("nft", data, lookback, horizon, num_epochs, blocks, series, 
                        #  train_mse, test_mse, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase)
    
    add_results_to_excel('nft', data, lookback, horizon, num_epochs, blocks, stack_types, 
                         fourier_granularity=thetas_dim[1], poly_degree=thetas_dim[0], 
                         num_channels=[25, 50], series=series, year=year, train_mse=train_mse, test_mse=test_mse, 
                         train_mae=train_mae, test_mae=test_mae, train_smape=train_smape, test_smape=test_smape, train_mape=train_mape,
                         test_mape=test_mape, train_mase=train_mase, test_mase=test_mase, train_rmsse=None, test_rmsse=None)
    
    # save_model(model, path_to_save_model, model_name)
    # print(f"path_to_save_model={path_to_save_model}, model_name={model_name}")


if __name__ == "__main__":
    data = 'noaa_years'
    print(f'data={data}')
    poly_degree,fourier_granularity = 4,8
    
    if data == 'seasonal_trend_test':
         for seasonal_amplitude in [0.05, 0.1, 0.15, 0.2]:
            for trend_amplitude in [30, 40, 50, 60]:
                data = f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
                for lookback, horizon in data_to_steps['seasonal_trend_0_5']:
                    if lookback==150 and horizon ==50: break      
                    main(
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=10,
                        plot_epoch=100,
                        blocks=data_to_num_nft_blocks.get(data, 2),
                        layers_type='tcn',
                        batch_size=32,
                        series=None,#'E00001', # "AEM00041194" #"AG000060590"
                        stack_types=('trend', 'seasonality'),
                        thetas_dim=(poly_degree, fourier_granularity),
                        get_forecast_and_coeffs=False,
                        dft_1d=False
                    )
    elif data == 'noaa_years':
        for series in single_data_to_series_list[data]:
            for year in noaa_series_to_years[series]:
                for lookback, horizon in data_to_steps[data]:
                    main(
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=10,
                        plot_epoch=100,
                        blocks=data_to_num_nft_blocks[data],
                        layers_type='tcn',
                        batch_size=32,
                        series=series,
                        year=year,
                        stack_types=('trend', 'seasonality'),
                        thetas_dim=(poly_degree, fourier_granularity),
                        get_forecast_and_coeffs=False,
                        dft_1d=False
                    )
    else:
        for lookback, horizon in data_to_steps[data]:
            main(
                data=data,
                lookback=lookback,
                horizon=horizon,
                num_epochs=10,
                plot_epoch=100,
                blocks=data_to_num_nft_blocks[data],
                layers_type='tcn',
                batch_size=32,
                series=None,#None,#'E00001', # "AEM00041194" #"AG000060590"
                year=None,
                stack_types=('trend', 'seasonality'),
                thetas_dim=(poly_degree, fourier_granularity),
                get_forecast_and_coeffs=False,
                dft_1d=False
            )