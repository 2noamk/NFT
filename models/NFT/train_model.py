import torch
import sys
import pytorch_lightning as pl

# sys.path.append('NFT/models/NFT/')
# from NFT_NFT import NFT

sys.path.append('NFT/models/NFT/')
from models.NFT.NFT import NFT


sys.path.append('NFT/')
from dicts import data_to_num_vars_dict, data_to_num_of_series

sys.path.append('NFT/')
from models.training_functions import get_data, evaluate_model, save_model, get_model_name, add_results_to_excel, get_path


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
):
    
    num_of_vars=data_to_num_vars_dict[data]    
    n_series=data_to_num_of_series[data]
    
    model_name = get_model_name('nft', data, lookback, horizon, num_epochs, blocks, layers_type, n_series, series)
    
    path_to_save_model, path_to_save_checkpoint_models, path_to_save_loss_plots, path_to_save_prediction_plots = get_path(model_name, data, 'nft')
    
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
        data=data, 
        lookback=lookback, 
        horizon=horizon,
        n_series=n_series,
        print_stats=False,
        series=series
        )

    model = NFT(
        forecast_length=horizon,
        backcast_length=lookback,
        n_vars=num_of_vars,
        nb_blocks_per_stack=blocks,
        layers_type=layers_type,
        num_channels_for_tcn=[25, 50]
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
        path_to_save_prediction_plots=path_to_save_prediction_plots
        )

    train_pred, val_pred, test_pred, train_mse, test_mse, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase = evaluate_model(model, train_X, train_y, val_X, val_y, test_X, test_y)

    add_results_to_excel("nft", data, lookback, horizon, num_epochs, blocks, series, train_mse, test_mse, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase)
    save_model(model, path_to_save_model, model_name)
    
    print(f"path_to_save_model={path_to_save_model}, model_name={model_name}")


if __name__ == "__main__":
    data = 'air_quality'
    lookback = 5
    horizon = 1
    main(
        data=data,
        lookback=lookback,
        horizon=horizon,
        num_epochs=1,
        plot_epoch=100,
        blocks=2,
        layers_type='tcn',
        batch_size=32,
        series=None, # "AEM00041194" #"AG000060590"
    )