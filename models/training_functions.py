import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from torch.utils.data import DataLoader
from pathlib import Path
from torch import nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_all_data_and_print_stats(data_path, print_stats=True):

    def read_pkl_to_torch(file_path, device=None, dtype=None):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        tensor_data = torch.tensor(data, device=device, dtype=dtype)
        return tensor_data

    def print_stats_func(train_X, train_y, val_X, val_y, test_X, test_y):
        print(f"shape of:\n"
        f"train: X {train_X.shape}, y: {train_y.shape}\n"
        f"val: X {val_X.shape}, y: {val_y.shape}\n"
        f"test: X {test_X.shape}, y: {test_y.shape}")
        print()

        print(f"The highest value in train is: {torch.max(train_X).item()}")
        print(f"The lowest value in the train is: {torch.min(train_X).item()}")
        print(f"The mean value in the train is: {torch.mean(train_X).item()}")
        print(f"The median value in the train is: {torch.median(train_X).item()}")
        print()

        print(f"The highest value in val is: {torch.max(val_X).item()}")
        print(f"The lowest value in the val is: {torch.min(val_X).item()}")
        print(f"The mean value in the val is: {torch.mean(val_X).item()}")
        print(f"The median value in the val is: {torch.median(val_X).item()}")
        print()

        print(f"The highest value in test is: {torch.max(test_X).item()}")
        print(f"The lowest value in test train is: {torch.min(test_X).item()}")
        print(f"The mean value in the test is: {torch.mean(test_X).item()}")
        print(f"The median value in the test is: {torch.median(test_X).item()}")

    train_X = read_pkl_to_torch(data_path + 'train_X.pkl', dtype=torch.float32)
    train_y = read_pkl_to_torch(data_path + 'train_y.pkl', dtype=torch.float32)
    val_X = read_pkl_to_torch(data_path + 'val_X.pkl', dtype=torch.float32)
    val_y = read_pkl_to_torch(data_path + 'val_y.pkl', dtype=torch.float32)
    test_X = read_pkl_to_torch(data_path + 'test_X.pkl', dtype=torch.float32)
    test_y = read_pkl_to_torch(data_path + 'test_y.pkl', dtype=torch.float32)
    
    if print_stats:
        print_stats_func(train_X, train_y, val_X, val_y, test_X, test_y)
    
    return train_X, train_y, val_X, val_y, test_X, test_y


def get_data(data, lookback, horizon, n_series, series=None, print_stats=True):    
    data_path=f"/home/noam.koren/multiTS/NFT/data/{data}/"

    if data in ['ecg', 'eeg']:
        data_path = data_path + f"{data}_{lookback}l_{horizon}h_{n_series}series/"
    elif data in ['eeg_single', 'ecg_single', 'noaa']:
        data_path = data_path + f"{series}/{series}_{lookback}l_{horizon}h/"
    else:
        data_path = data_path + f"{data}_{lookback}l_{horizon}h/"

    train_X, train_y, val_X, val_y, test_X, test_y = read_all_data_and_print_stats(
        data_path=data_path,
        print_stats=print_stats
        )

    return  train_X, train_y, val_X, val_y, test_X, test_y


def plot_loss_over_epoch_graph(epoch, train_losses, val_losses, lookback, horizon, path_to_save_loss_plots=None): 
    plt.figure(figsize=(3, 2))
    plt.plot(range(0, epoch), train_losses, label='Train Loss')
    plt.plot(range(0, epoch), val_losses, label='Validation Loss')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Train and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if path_to_save_loss_plots is not None:
        plt.savefig(path_to_save_loss_plots + f"/loss_epoch={epoch}.png")
    plt.show()


def plot_func(actual_series=None, predicted_series=None, n_vars=None, epoch=None, path_to_save_prediction_plots=None):
    n_cols = 4
    n_rows = int(n_vars / n_cols) + 1
    plt.figure(figsize=(50, 25))

    for series in range(n_vars):
        plt.subplot(n_rows, n_cols, series + 1)
        if actual_series is not None: plt.plot(actual_series[:, series], label="Actual")
        if predicted_series is not None: plt.plot(predicted_series[:, series], label="Predicted")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Series {series + 1}")

    plt.suptitle("Actual vs Predicted Time Series")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # To ensure the title fits
    if path_to_save_prediction_plots is not None:
        plt.savefig(path_to_save_prediction_plots + f"/predictions_epoch={epoch}.png")
    plt.show()


def plot_specific_prediction(y=None, predictions=None, idx=None, horizon=None, n_vars=None, epoch=None, path_to_save_prediction_plots=None):
    def get_series(s):
        if s is not None:
            if not isinstance(s, np.ndarray): s = s.cpu().numpy()
            if idx is not None: s = s.reshape(-1, horizon, n_vars)[idx, :, :]
        return s
    
    actual_series = get_series(y)
    predicted_series = get_series(predictions)
    
    plot_func(actual_series, predicted_series, n_vars, epoch, path_to_save_prediction_plots)
    

def plot_predictions(y=None, predictions=None, horizon=None, n_vars=None, epoch=None, path_to_save_prediction_plots=None):
    """
    function takes the first time stamp predicted for every sample and plots it.
    i.e; function does not plot later horizon points.
    """
    
    def get_series(s):
        if s is not None:
            if not isinstance(s, np.ndarray): s = s.cpu().numpy()
            s = s.reshape(-1, horizon, n_vars)[:, 0, :]
        return s
    
    actual_series = get_series(y)
    predicted_series = get_series(predictions)
    
    plot_func(actual_series, predicted_series, n_vars, epoch, path_to_save_prediction_plots)


def plot_poly(trend_thetas):
    coeffs_np = trend_thetas.mean(dim=0).cpu().numpy()
    
    _, horizon = coeffs_np.shape

    x_values = np.linspace(0, 1, horizon)

    plt.figure(figsize=(15, 10))
    for i, coef in enumerate(coeffs_np):
        y_values = np.polyval(coef[::-1], x_values)
        plt.plot(x_values, y_values, label=f'Poly {i+1}')

    plt.title('Polynomials of Degree 4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    return


def plot_fourier(seasonality_thetas):
    fourier_coeffs_np = seasonality_thetas.mean(dim=0).cpu().numpy()
    _, horizon = fourier_coeffs_np.shape
    x_values = np.linspace(0, 1, horizon)
    
    def fourier_series(x, coeffs):
        n_terms = len(coeffs) // 2
        result = np.zeros_like(x)
        for i in range(n_terms):
            result += coeffs[i] * np.cos(2 * np.pi * (i + 1) * x)
            result += coeffs[i + n_terms] * np.sin(2 * np.pi * (i + 1) * x)
        return result

    plt.figure(figsize=(15, 10))
    for i, coef in enumerate(fourier_coeffs_np):
        y_values = fourier_series(x_values, coef)
        plt.plot(x_values, y_values, label=f'Fourier Series {i+1}')

    plt.title('Fourier Series')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
  
   
def save_predictions_in_excel(y=None, predictions=None, horizon=None, n_vars=None, path_to_save_prediction_excel=None):
    """
    function takes the first time stamp predicted for every sample and saves it in an Excel file.
    i.e; function does not save later horizon points.
    """
    def reshape_series(s):
        if s is not None: 
            if not isinstance(s, np.ndarray): s.cpu().numpy()
            s = s.reshape(-1, horizon, n_vars)[:, 0, :]
        return s

    actual_series = reshape_series(y)
    predicted_series = reshape_series(predictions)

    data = {}
    for series in range(n_vars):
        if y is not None:
            data[f'Actual Series {series + 1}'] = actual_series[:, series]
        if predictions is not None:
            data[f'Predicted Series {series + 1}'] = predicted_series[:, series]

    df = pd.DataFrame(data)

    if path_to_save_prediction_excel is not None:
        excel_filename = path_to_save_prediction_excel + f"predictions.xlsx"
        df.to_excel(excel_filename, index=False)

    return df 


def train_model(dataset, epochs, model, train_X, train_y, val_X, val_y, lookback, horizon, n_vars=1, batch_size=32, print_epoch=10, path_to_save_prediction_plots=None, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = dataset(train_X, train_y)
    if val_X != None: val_dataset = dataset(val_X, val_y)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    if val_X != None: val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    min_train_loss = min_val_loss = float('inf')
    min_train_loss_epoch = min_val_loss_epoch = 0

    for epoch in range(epochs):
        val_predictions = []
        
        model.train()
        total_loss_sum = 0.0
        total_data_points = 0
        for batch_x, y_target in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_x.to(device))
            y_pred = output[1] if isinstance(output, tuple) else output
            loss = criterion(y_pred.to(device), y_target.to(device))
            loss.backward()
            optimizer.step()
            total_loss_sum += loss.item() * len(batch_x)
            total_data_points += len(batch_x)
        train_loss = (total_loss_sum / total_data_points) # / (horizon * n_vars)
        if train_loss < min_train_loss: 
            min_train_loss = train_loss
            min_train_loss_epoch = epoch
        if epoch != 0: train_losses.append(train_loss)
        
        if val_X != None:
            model.eval()
            total_val_loss = 0.0
            total_data_points = 0
            with torch.no_grad():
                for batch_x, y_target in val_dataloader:
                    output = model(batch_x.to(device))
                    y_pred = output[1] if isinstance(output, tuple) else output
                    loss = criterion(y_pred.to(device), y_target.to(device))
                    total_val_loss += loss.item() * len(batch_x)
                    total_data_points += len(batch_x)
                    val_predictions.append(y_pred.cpu().numpy())  # Save the predictions
            
            val_loss = (total_val_loss/total_data_points)#/ (horizon * n_vars)
            val_predictions = np.concatenate(val_predictions, axis=0)  # Stacks along the first axis (axis=0) 
            if val_loss < min_val_loss: 
                min_val_loss = val_loss
                min_val_loss_epoch = epoch
        else:
            val_loss = float('inf')
            val_predictions = None
        if epoch != 0: val_losses.append(val_loss)
        
        print(f'Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs}, loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
            
        if (epoch % print_epoch == 0 or epoch == epochs - 1) and epoch != 0:
            plot_predictions(
                y=val_y, 
                predictions=val_predictions,
                horizon=horizon, 
                n_vars=n_vars,
                epoch=epoch, 
                path_to_save_prediction_plots=path_to_save_prediction_plots
                )
            plot_loss_over_epoch_graph(
                epoch, 
                train_losses, 
                val_losses, 
                lookback=lookback,
                horizon=horizon,
                path_to_save_loss_plots=None
                )
    
    print(f"The min train loss is {min_train_loss} at epoch {min_train_loss_epoch}\n"
        f"The min val loss is {min_val_loss} at epoch {min_val_loss_epoch}\n")  
                       
    return train_loss, val_loss, train_dataset, val_dataset, val_predictions   


def calculate_smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE)
    :param y_true: The actual values
    :param y_pred: The predicted values
    :return: sMAPE
    """
    y_true, y_pred = np.array(y_true.cpu()), np.array(y_pred.cpu())
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # handle the case where the denominator is zero
    return 100 * np.mean(diff)


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE)
    :param y_true: The actual values
    :param y_pred: The predicted values
    :return: MAPE
    """
    y_true, y_pred = np.array(y_true.cpu()), np.array(y_pred.cpu())
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                        y_true[non_zero_mask])) * 100


def calculate_mase(y_true, y_pred):
    """
    Calculate the Mean Absolute Scaled Error (MASE)
    :param y_true: The actual values
    :param y_pred: The predicted values
    :return: MASE
    """
    y_true, y_pred = np.array(y_true.cpu()), np.array(y_pred.cpu())
    # Generate y_naive, where each value is the preceding value in y_true
    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]  # Assuming the first value is the same, or you can exclude it

    mae_pred = np.mean(np.abs(y_pred - y_true))
    mae_naive = np.mean(np.abs(y_naive - y_true))
    return mae_pred / mae_naive if mae_naive != 0 else np.inf


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
        f"train MAPE = {train_mase}\n"
        f"val MAPE = {val_mase}\n"
        f"test MAPE = {test_mase}\n")
    
    model.eval()

    train_pred = model.predict(train_X).clone().detach() # torch.tensor(tcn_model.predict(train_X), device=device).float()
    val_pred = model.predict(val_X).clone().detach() # torch.tensor(tcn_model.predict(val_X), device=device).float()
    test_pred = model.predict(test_X).clone().detach() # torch.tensor(tcn_model.predict(test_X), device=device).float()

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


def add_results_to_excel(model, data, lookback, horizon, epochs, blocks, series, train_mse, test_mse, train_smape, test_smape, train_mape, test_mape, train_mase, test_mase):
        
    base_dir = f"/home/noam.koren/multiTS/NFT/results/{data}/{model}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if series is not None:
        data = f"{data}_{series}"
        
    file_path = f"{base_dir}/{model}_{data}_results.xlsx"
        
    if Path(file_path).is_file(): df = pd.read_excel(file_path)
    else: df = pd.DataFrame(columns=["Data", "Lookback", "Horizon", "Epochs", "Blocks",
                                   "train_mse", "test_mse", "train_smape", "test_smape", 
                                   "train_mape", "test_mape", "train_mase", "test_mase"])
    
    if torch.is_tensor(train_mse): train_mse = train_mse.cpu().item()
    if torch.is_tensor(test_mse): test_mse = test_mse.cpu().item()
        
    new_row = {"Data": data, "Lookback": lookback, "Horizon": horizon, 
               "Epochs": epochs, "Blocks": blocks,
               "train_mse": train_mse, "test_mse": test_mse, 
               "train_smape": train_smape, "test_smape": test_smape, 
               "train_mape": train_mape, "test_mape": test_mape, 
               "train_mase": train_mase, "test_mase": test_mase}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(file_path, index=False)


def save_model(model, path_to_save_model, model_name):
    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)
    torch.save(model.state_dict(), f"{path_to_save_model}{model_name}.pth")


def get_model_name(model_type, data, lookback, horizon, epochs, blocks, layers_type, n_series, series=None):
    model_name = f"{model_type}_{lookback}l_{horizon}h_{epochs}epochs_{blocks}blocks_{layers_type}"
    if data == 'ecg':
        model_name += f"_{n_series}series"
    elif data == 'ecg_single' or data == 'noaa':
        model_name += f"_{series}"
    return model_name


def get_path(model_name, data, model_type):
    trained_models_path=f"/home/noam.koren/multiTS/NFT/models/trained_models/"
    path_to_save_model = trained_models_path + f"{data}/{model_type}/{model_name}/"    
    path_to_save_checkpoint_models = path_to_save_model + "models_during_train/"
    path_to_save_loss_plots = path_to_save_model + "loss_plots/"
    path_to_save_prediction_plots = path_to_save_model + "prediction_plots/"
    
    return path_to_save_model, path_to_save_checkpoint_models, path_to_save_loss_plots, path_to_save_prediction_plots
    

def log_to_file(message, file_path):
    with open(file_path, 'a') as file:
        file.write(message + '\n')


def print_conclusion_to_txt(
    out_txt_name,
    model_type,
    data,
    num_epochs,
    blocks,
    lookback,
    horizon,
    train_mse,
    val_mse,
    test_mse,
    train_metrics,
    val_metrics,
    test_metrics
):
    
    txt_path = f"/home/noam.koren/multiTS/NFT/models/NFT/{out_txt_name}.txt"

    log_config = f"model = {model_type}, data = {data}, epochs = {num_epochs},blocks = {blocks}, lookback = {lookback}, horizon = {horizon}"
    log_to_file(log_config, txt_path)
    
    log_mse = f"Training Loss: {train_mse}\nValidation Loss: {val_mse}\nTest Loss: {test_mse}"
    log_to_file(log_mse, txt_path)
    
    log_metrics = f"Training Metrics: {train_metrics}\nValidation Metrics: {val_metrics}\nTest Metrics: {test_metrics}"
    log_to_file(log_metrics, txt_path)
