import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.nn import functional as F

models_path = "/home/../multiTS/NFT"
if models_path not in sys.path: sys.path.append(models_path)

from models.training_functions import train_model, calculate_smape, calculate_mape, calculate_mase
from models.baseline_models.base_models import NBeatsNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataset(Dataset):
    def __init__(self, data, target):
        self.input = data
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx): 
        return self.input[idx], self.target[idx]  

def evaluate_model(model, train_X, train_y, val_X, val_y, test_X, test_y):
    def print_evaluation(train_mse, val_mse, test_mse,
                        train_smape, val_smape, test_smape, 
                        train_mape, val_mape, test_mape,
                        train_mase, val_mase, test_mase):
                                    
        print(f"MSE:\n"
        f"train MSE = {train_mse}\n"
        f"val MSE = {val_mse}\n"
        f"test MSE = {test_mse}\n")
        
        print(f"sMAPE:\n"
        f"train sMAPE = {train_smape}\n"
        f"val sMAPE = {val_smape}\n"
        f"test sMAPE = {test_smape}\n")
        
        print(f"MAPE:\n"
        f"train MAPE = {train_mape}\n"
        f"val MAPE = {val_mape}\n"
        f"test MAPE = {test_mape}\n")
        
        print(f"MASE:\n"
        f"train MASE = {train_mase}\n"
        f"val MASE = {val_mase}\n"
        f"test MASE = {test_mase}\n")
    
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

    train_smape = calculate_smape(train_pred, train_y)
    val_smape = calculate_smape(val_pred, val_y)
    test_smape = calculate_smape(test_pred, test_y)
    
    train_mape = calculate_mape(train_pred, train_y)
    val_mape = calculate_mape(val_pred, val_y)
    test_mape = calculate_mape(test_pred, test_y)
    
    train_mase = calculate_mase(train_pred, train_y)
    val_mase = calculate_mase(val_pred, val_y)
    test_mase = calculate_mase(test_pred, test_y)
    
    print_evaluation(train_mse, val_mse, test_mse,
                     train_smape, val_smape, test_smape, 
                     train_mape, val_mape, test_mape,
                     train_mase, val_mase, test_mase)    
    return train_pred, val_pred, test_pred, train_mse, test_mse, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase

def get_data(lookback, horizon, n_vars, plot=True):
    def create_datasets(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[(i + lookback):(i + lookback + horizon)])
        return np.array(X), np.array(y)

    def split_data(X, y):
        n_samples = X.shape[0]
        train_samples = int(n_samples * 0.7)
        val_samples = int(n_samples * 0.15)

        train_X = torch.tensor(X[:train_samples], dtype=torch.float32)
        train_y = torch.tensor(y[:train_samples], dtype=torch.float32)

        val_X = torch.tensor(X[train_samples:train_samples + val_samples], dtype=torch.float32)
        val_y = torch.tensor(y[train_samples:train_samples + val_samples], dtype=torch.float32)

        test_X = torch.tensor(X[train_samples + val_samples:], dtype=torch.float32)
        test_y = torch.tensor(y[train_samples + val_samples:], dtype=torch.float32)
        
        return train_X.to(device), train_y.to(device), val_X.to(device), val_y.to(device), test_X.to(device), test_y.to(device)

    n_points = 365 * 4 
    time = np.arange(n_points)


    polynomial1 = 1 * time**3 + 2 * time**2 + 3 * time + 4

    polynomial2 = 0 * time**3 + 10 * time**2 + 5 * time + 0

    polynomial3 = -5 * time**3 - 15 * time**2 + 5 * time + 3

    polynomial4 = -0.7 * time**3 + 6 * time**2 -0.7 * time - 7

    polynomial5 = 3 * time**3 - 27 * time**2 - 0.5 * time + 2.1

    polynomial6 = 4 * time**3 + 2 * time**2 + 1 * time + 1

    polynomial7 = -4 * time**3 - 0.1 * time**2 - 7 * time + 25

    polynomial8 = 0.4 * time**3 - 6 * time**2 - 0.1 * time - 6

    p = (polynomial1, polynomial2, polynomial3, polynomial4, polynomial5, polynomial6, polynomial7, polynomial8)


    multivariate_time_series = np.column_stack(p[:n_vars])
    
    if plot:
        plt.figure(figsize=(12, n_vars))
        for i in range(n_vars): plt.plot(time, p[i], label=f'Polynomial {i}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Polynomials Over Time')
        plt.legend()
        plt.show()

    print(f"multivariate_time_series.shape={multivariate_time_series.shape}")

    X, y = create_datasets(multivariate_time_series, lookback, horizon)

    train_X, train_y, val_X, val_y, test_X, test_y = split_data(X, y)

    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)
    
    return train_X, train_y, val_X, val_y, test_X, test_y

def to_numpy(data):
    """Converts input data to a NumPy array, handling both PyTorch tensors and NumPy arrays."""
    if isinstance(data, torch.Tensor):
        # Move tensor to CPU if it's on a CUDA device, then convert to NumPy
        return data.cpu().numpy() if data.is_cuda else data.numpy()
    elif isinstance(data, np.ndarray):
        # If it's already a NumPy array, just return it
        return data
    else:
        # If it's a scalar (e.g., Python int or float), convert to a NumPy array
        return np.array(data)

def train(
    lookback, 
    horizon, 
    stack_types, 
    thetas_dim, 
    nb_blocks_per_stack, 
    epoch, 
    file_path,
    n_vars=8,
):
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
    lookback=lookback, 
    horizon=horizon,
    n_vars=n_vars,
    plot=False
    )

    mse_lst, smape_lst, mase_lst = [], [], []
    

    for idx in range(n_vars):
        print(f"idx={idx}, train_X={train_X.shape}")

        nbeats_model = NBeatsNet(
        stack_types=stack_types,
        forecast_length=horizon,
        backcast_length=lookback,
        nb_blocks_per_stack=nb_blocks_per_stack,
        thetas_dim=thetas_dim,
        ).to(device)
        
        train_X_idx, train_y_idx = train_X[:,:,idx], train_y[:,:,idx]
        val_X_idx, val_y_idx = val_X[:,:,idx], val_y[:,:,idx]
        test_X_idx, test_y_idx = test_X[:,:,idx], test_y[:,:,idx]
        
        print(f"train_y_idx, val_y_idx, test_y_idx={train_y_idx.shape, val_y_idx.shape, test_y_idx.shape}")
        
        train_model(
                dataset=dataset, 
                epochs=epoch, 
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
        
        train_pred, val_pred, test_pred, train_mse, test_mse, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase = evaluate_model(nbeats_model, train_X_idx, train_y_idx, val_X_idx, val_y_idx, test_X_idx, test_y_idx)


        mse_lst.append(test_mse.item() if isinstance(test_mse, torch.Tensor) else test_mse)
        mase_lst.append(test_mase.item() if isinstance(test_mase, torch.Tensor) else test_mase)
        smape_lst.append(test_smape.item() if isinstance(test_smape, torch.Tensor) else test_smape)

    mse_vals = [np.mean(mse_lst[:i]) for i in range(1, 9)]    
    smape_vals = [np.mean(smape_lst[:i]) for i in range(1, 9)]
    mase_vals = [np.mean(mase_lst[:i]) for i in range(1, 9)]
    
    for i in range(n_vars):

        if Path(file_path).is_file(): df = pd.read_excel(file_path)
        else: df = pd.DataFrame(columns=["Lookback", "Horizon", "Vars", "Epochs", "Stacks", 
                                        "Dagree", "Blocks", "Is Poly", 
                                    "test_mse", "test_smape", "test_mase"])

        new_row = {"Lookback": lookback, "Horizon": horizon, "Vars": n_vars,
                "Epochs": epochs,  "Stacks": stack_types, 
                "Dagree":thetas_dim, "Blocks": nb_blocks_per_stack, 
                "Is Poly":'Nbeats', "test_mse": to_numpy(mse_vals[i]), 
                "test_smape": to_numpy(smape_vals[i]), "test_mase": to_numpy(mase_vals[i])}
    
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_excel(file_path, index=False)

if __name__ == '__main__':
    plot = False

    steps = [(7, 1), (15, 1), (15, 7), (30, 1), (30, 7), (30, 15)]
    n_vars = [i for i in range(1, 9)]

    stacks = [ 
              (('trend', 'seasonality'), [(1,8), (2,8), (3,8), (4,8)]), 
              (('trend', 'trend'), [(1, 1), (2,2), (3,3), (4,4)]),
              (('seasonality', 'seasonality'), [(8,8), (8,8), (8,8), (8,8)]),
            ]
    blocks = [1,2,3]
    epochs = [15]
    
        
    excel_path = '/home/../multiTS/NFT/models/tests/test_trend/evaluate_nbeats_trend_on_syntatic_data.xlsx'

    for s in steps:
        lookback, horizon = s
        for st in stacks:
            stack_types, thetas = st
            for thetas_dim in thetas:
                for nb_blocks_per_stack in blocks:
                    for epoch in epochs:
                        if stack_types == ('seasonality', 'seasonality'):
                            train(lookback, horizon, stack_types, thetas_dim, nb_blocks_per_stack, epoch, excel_path)
                        else:
                            train(lookback, horizon, stack_types, thetas_dim, nb_blocks_per_stack, epoch, excel_path)
                            train(lookback, horizon, stack_types, thetas_dim, nb_blocks_per_stack, epoch, excel_path)