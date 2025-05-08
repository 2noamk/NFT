import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

models_path = "/home/../multiTS/NFT"
if models_path not in sys.path: sys.path.append(models_path)

from models.NFT.NFT import NFT
from models.training_functions import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(lookback, horizon, n_vars, plot=True, sinusodial=False):
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
    
    if sinusodial:
        polynomial1 = polynomial1 + 100000000 * np.sin(3 * time)
        polynomial2 = polynomial2 + 50000000 * np.sin(1 * time)
        polynomial3 = polynomial3 + 300000000 * np.sin(3 * time)
        polynomial4 = polynomial4 + 100000000 * np.sin(3 * time)
        polynomial5 = polynomial5 + 10000000 * np.sin(4 * time)
        polynomial6 = polynomial6 + 900000000 * np.sin(0.02 * time)
        polynomial7 = polynomial7 + 50000000 * np.sin(1 * time)
        polynomial8 = polynomial8 + 50000000 * np.sin(0.7 * time)

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

def train_model_and_save_evals(lookback, horizon, n_vars, stack_types, thetas_dim, nb_blocks_per_stack, epochs, train_X, train_y, val_X, val_y, test_X, test_y, is_poly):
    model = NFT(
        forecast_length=horizon,
        backcast_length=lookback,
        stack_types=stack_types,
        thetas_dim=thetas_dim,
        n_vars=n_vars,
        nb_blocks_per_stack=nb_blocks_per_stack,
        num_channels_for_tcn=[2, 2],
        is_poly=is_poly
        ).to(device)

    model.compile(loss='mse', optimizer='adam')

    model.fit(
        x_train=train_X, 
        y_train=train_y, 
        validation_data=(val_X, val_y), 
        epochs=epochs,
        batch_size=32, 
        plot_epoch=100
        )

    _, _, _, _, test_mse, _, test_smape, _, _, _, test_mase = evaluate_model(model, train_X, train_y, val_X, val_y, test_X, test_y)
    
    file_path = '/home/../multiTS/NFT/models/tests/test_trend/evaluate_sinusodial_trend_on_syntatic_data.xlsx'

    if Path(file_path).is_file(): df = pd.read_excel(file_path)
    else: df = pd.DataFrame(columns=["Lookback", "Horizon", "Vars", "Epochs", "Stacks", 
                                     "Dagree", "Blocks", "Is Poly", 
                                   "test_mse", "test_smape", "test_mase"])

    new_row = {"Lookback": lookback, "Horizon": horizon, "Vars": n_vars,
               "Epochs": epochs,  "Stacks": stack_types, 
               "Dagree":thetas_dim, "Blocks": nb_blocks_per_stack, 
               "Is Poly":is_poly, "test_mse": to_numpy(test_mse), 
               "test_smape": to_numpy(test_smape), "test_mase": to_numpy(test_mase)}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(file_path, index=False)

if __name__ == '__main__':
    plot = False

    steps = [(60, 1), (60, 10), (60, 15), (60, 30),
             (90, 1), (90, 10), (90, 15), (90, 30),
             (120, 1), (120, 10), (120, 15), (120, 30),
             ]
    n_vars = [i for i in range(1, 9)]

    stacks = [ 
              (('trend', 'seasonality'), [(1,8), (2,8), (3,8), (4,8)]), 
            #   (('seasonality', 'seasonality'), [(8,8), (8,8), (8,8), (8,8)]),
            #   (('trend', 'trend'), [(1, 1), (2,2), (3,3), (4,4)]),
            ]
    blocks = [1,2,3]
    epochs = [15]
    
    sinusodial=True

    for s in steps:
        lookback, horizon = s
        for v in n_vars:
            train_X, train_y, val_X, val_y, test_X, test_y =  get_data(lookback, horizon, v, plot, sinusodial=sinusodial)
            for st in stacks:
                stack_types, thetas = st
                for thetas_dim in thetas:
                    for nb_blocks_per_stack in blocks:
                        for epoch in epochs:
                            if stack_types == ('seasonality', 'seasonality'):
                                train_model_and_save_evals(lookback, horizon, v, stack_types, thetas_dim, nb_blocks_per_stack, epoch, train_X, train_y, val_X, val_y, test_X, test_y, is_poly=False)
                            else:
                                train_model_and_save_evals(lookback, horizon, v, stack_types, thetas_dim, nb_blocks_per_stack, epoch, train_X, train_y, val_X, val_y, test_X, test_y, is_poly=True)
                                train_model_and_save_evals(lookback, horizon, v, stack_types, thetas_dim, nb_blocks_per_stack, epoch, train_X, train_y, val_X, val_y, test_X, test_y, is_poly=False)