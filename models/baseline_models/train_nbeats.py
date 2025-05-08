import torch
from torch.nn import functional as F
import sys
from torch.utils.data import Dataset
import numpy as np

sys.path.append('/home/../multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_num_of_series, data_to_steps, single_data_to_series_list
from models.training_functions import get_data, train_model, evaluate_model, calculate_smape, calculate_mape, calculate_mase, calculate_mae, add_results_to_excel
from models.baseline_models.base_models import NBeatsNet, FC, LSTM, CNN, TCN, TimeSeriesTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, train_X, train_y, val_X, val_y, test_X, test_y):
    # def print_evaluation(train_mse, val_mse, test_mse,
    #                     train_smape, val_smape, test_smape, 
    #                     train_mape, val_mape, test_mape,
    #                     train_mase, val_mase, test_mase):
                                    
    #     print(f"MSE:\n"
    #     f"train MSE = {train_mse}\n"
    #     f"val MSE = {val_mse}\n"
    #     f"test MSE = {test_mse}\n")
        
    #     print(f"sMAPE:\n"
    #     f"train sMAPE = {train_smape}\n"
    #     f"val sMAPE = {val_smape}\n"
    #     f"test sMAPE = {test_smape}\n")
        
    #     print(f"MAPE:\n"
    #     f"train MAPE = {train_mape}\n"
    #     f"val MAPE = {val_mape}\n"
    #     f"test MAPE = {test_mape}\n")
        
    #     print(f"MASE:\n"
    #     f"train MASE = {train_mase}\n"
    #     f"val MASE = {val_mase}\n"
    #     f"test MASE = {test_mase}\n")
    
    model.eval()

    if isinstance(train_X, np.ndarray): train_X = torch.from_numpy(train_pred)
    if isinstance(val_X, np.ndarray): val_X = torch.from_numpy(val_pred)
    if isinstance(test_X, np.ndarray): test_X = torch.from_numpy(test_pred)
    
    train_pred = model.predict(train_X.to(device))
    val_pred = model.predict(val_X.to(device))
    test_pred = model.predict(test_X.to(device))
    
    if isinstance(train_pred, np.ndarray): train_pred = torch.from_numpy(train_pred)
    if isinstance(val_pred, np.ndarray): val_pred = torch.from_numpy(val_pred)
    if isinstance(test_pred, np.ndarray): test_pred = torch.from_numpy(test_pred)

    train_mse = F.mse_loss(train_pred.to(device), train_y.to(device))
    val_mse = F.mse_loss(val_pred.to(device), val_y.to(device))
    test_mse = F.mse_loss(test_pred.to(device), test_y.to(device))

    # Calculate MAE loss
    train_mae = calculate_mae(train_pred, train_y)
    val_mae = calculate_mae(train_pred, train_y)
    test_mae = calculate_mae(train_pred, train_y)

    train_smape = calculate_smape(train_pred, train_y)
    val_smape = calculate_smape(val_pred, val_y)
    test_smape = calculate_smape(test_pred, test_y)
    
    train_mape = calculate_mape(train_pred, train_y)
    val_mape = calculate_mape(val_pred, val_y)
    test_mape = calculate_mape(test_pred, test_y)
    
    train_mase = calculate_mase(train_pred, train_y)
    val_mase = calculate_mase(val_pred, val_y)
    test_mase = calculate_mase(test_pred, test_y)
    
    # print_evaluation(train_mse, val_mse, test_mse,
                    #  train_smape, val_smape, test_smape, 
                    #  train_mape, val_mape, test_mape,
                    #  train_mase, val_mase, test_mase)    
    return train_pred, val_pred, test_pred, train_mse, test_mse, train_mae, test_mae, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase


class dataset(Dataset):
    def __init__(self, data, target):
        self.input = data
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx): 
        return self.input[idx], self.target[idx]
  

def train(
    data,
    lookback,
    horizon,
    epochs,
    blocks,
    n_series=600,
    series=None
):
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
    data=data, 
    lookback=lookback, 
    horizon=horizon,
    n_series=n_series,
    series=series,
    print_stats=False
    )
    

    mse_lst, smape_lst, mase_lst, mae_lst = [], [], [], []
    
    n_vars = data_to_num_vars_dict[data]

    for idx in range(n_vars):
        print(f"idx={idx}")
        print(f"train_X={train_X.shape}")

        nbeats_model = NBeatsNet(
        # stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
        forecast_length=horizon,
        backcast_length=lookback,
        nb_blocks_per_stack=blocks,
        thetas_dim=(8,4,8),
        ).to(device)
        train_X_idx, train_y_idx = train_X[:,:,idx], train_y[:,:,idx]
        val_X_idx, val_y_idx = val_X[:,:,idx], val_y[:,:,idx]
        test_X_idx, test_y_idx = test_X[:,:,idx], test_y[:,:,idx]
        print(f"train_X={train_X_idx.shape}")
        train_model(
                dataset=dataset, 
                epochs=epochs, 
                model=nbeats_model, 
                train_X=train_X_idx.to(device), train_y=train_y_idx.to(device),
                val_X=val_X_idx.to(device), val_y=val_y_idx.to(device), 
                device=device, 
                lookback=lookback,
                horizon=horizon, 
                n_vars=n_vars, 
                batch_size=32, 
                print_epoch=epochs, 
                path_to_save_prediction_plots=None)
        
        print(f"train_y_idx, val_y_idx, test_y_idx={train_y_idx.shape, val_y_idx.shape, test_y_idx.shape}")
        train_pred, val_pred, test_pred, train_mse, test_mse, train_mae, test_mae, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase = evaluate_model(nbeats_model, train_X_idx, train_y_idx, val_X_idx, val_y_idx, test_X_idx, test_y_idx)


        mse_lst.append(test_mse.item() if isinstance(test_mse, torch.Tensor) else test_mse)
        mae_lst.append(test_mae.item() if isinstance(test_mae, torch.Tensor) else test_mae)
        mase_lst.append(test_mase.item() if isinstance(test_mase, torch.Tensor) else test_mase)
        smape_lst.append(test_smape.item() if isinstance(test_smape, torch.Tensor) else test_smape)
        

    mse = np.mean(mse_lst)
    mae = np.mean(mae_lst)
    smape = np.mean(smape_lst)
    mase = np.mean(mase_lst)
    
    print(f"mse = {mse}, smape = {smape}, mase = {mase}")

    add_results_to_excel(
        model='nbeats', 
        data=data, 
        num_channels=None,
        stack_types=None,
        lookback=lookback, 
        horizon=horizon, 
        epochs=epochs, 
        blocks=blocks, 
        series=series, 
        train_mse=0, 
        test_mse=mse,  
        train_mae=0, 
        test_mae=mae,   
        train_rmsse=0, 
        test_rmsse=0, 
        train_smape=0, 
        test_smape=smape, 
        train_mape=0, 
        test_mape=0, 
        train_mase=0, 
        test_mase=mase
        )
    
  
def main():
    data = 'noaa'
    epochs = [10]
    blocks = 2
    print(f"data = {data}")
    
    if data in ['eeg_single', 'ecg_single', 'noaa']:
        for series in single_data_to_series_list[data]:
            print(f"series={series}")
            for s in data_to_steps[data]:
                for num_epochs in epochs:
                    lookback, horizon = s
                    train(
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        epochs=num_epochs,
                        blocks=blocks,
                        series=series,
                    )
    else:
        for s in data_to_steps[data]:
            for num_epochs in epochs:
                lookback, horizon = s
                train(
                    data=data,
                    lookback=lookback,
                    horizon=horizon,
                    epochs=num_epochs,
                    blocks=blocks
                )
        



if __name__ == '__main__':
    main()