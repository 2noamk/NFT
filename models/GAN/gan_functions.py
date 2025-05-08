import os
import time
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
BASE_PATH = "/home/../multiTS/NFT/"


class Noiser(nn.Module):
    """Learns to generate adversarial noise."""
    def __init__(self, input_dim, hidden_dim=32):
        super(Noiser, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class FourierNoise(nn.Module):
    def __init__(self, num_of_vars, hidden_dim=32):
        super(FourierNoise, self).__init__()
        self.adversarial_layer = Noiser(num_of_vars, hidden_dim)

    def forward(self, x):
            # Convert to 2D Fourier space
            x_freq = torch.fft.fft2(x)  # 2D DFT

            # Extract real and imaginary parts
            real_part = torch.real(x_freq)
            imag_part = torch.imag(x_freq)

            # Modify only the real part
            real_part_modified = self.adversarial_layer(real_part)

            # Reconstruct modified Fourier coefficients
            x_freq_modified = real_part_modified + 1j * imag_part

            # Convert back to the original space
            x_adv = torch.fft.ifft2(x_freq_modified).real  # Take only the real part

            return x_adv

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target):
        self.input = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx): 
        return self.input[idx], self.target[idx]


def evaluate_forecaster(forecaster, dataloader, noise_scale=0.05, noiser=None):
    """Evaluates the forecaster on clean and noisy test data."""
    forecaster.eval()
    if noiser:
        noiser.eval()

    total_mse, total_mae, count = 0, 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_criterion, mae_criterion = nn.MSELoss(), nn.L1Loss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if noiser:
                x = x + noise_scale * noiser(x)

            y_pred = forecaster(x)
            total_mse += mse_criterion(y_pred, y).item()
            total_mae += mae_criterion(y_pred, y).item()
            count += 1

    return total_mse / count, total_mae / count

def train_gan_forecaster(train_loader, test_loader, forecaster, noiser, 
                         optimizer_forecaster, optimizer_noiser, 
                         criterion, noise_scale=0.1, num_epochs=10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster.to(device)
    noiser.to(device)

    for epoch in range(num_epochs):
        total_loss_forecaster, total_loss_noiser = 0, 0
        num_batches = len(train_loader)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Train Noiser
            optimizer_noiser.zero_grad()
            x_noisy = x + noise_scale * noiser(x)
            loss_noiser = -criterion(forecaster(x_noisy), y)
            loss_noiser.backward()
            optimizer_noiser.step()

            # Train Forecaster
            optimizer_forecaster.zero_grad()
            loss_forecaster = criterion(forecaster(x), y) + criterion(forecaster(x_noisy.detach()), y)
            loss_forecaster.backward()
            optimizer_forecaster.step()

            total_loss_forecaster += loss_forecaster.item()
            total_loss_noiser += loss_noiser.item()

        # Compute average training loss
        avg_loss_forecaster = total_loss_forecaster / num_batches
        avg_loss_noiser = total_loss_noiser / num_batches

        # Evaluate the model on test data (clean & noisy)
        clean_test_mse, clean_test_mae = evaluate_forecaster(forecaster, test_loader, noiser=None)
        noisy_test_mse, noisy_test_mae = evaluate_forecaster(forecaster, test_loader, noise_scale=noise_scale, noiser=noiser)

        # Print results for this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Forecaster Loss: {avg_loss_forecaster:.4f}, "
              f"Train Noiser Loss: {avg_loss_noiser:.4f}, "
              f"Test MSE (Clean): {clean_test_mse:.4f}, MAE (Clean): {clean_test_mae:.4f}, "
              f"Test MSE (Noisy): {noisy_test_mse:.4f}, MAE (Noisy): {noisy_test_mae:.4f}")

    return forecaster, noiser


def add_gan_results_to_excel(model, data, **kwargs):
    directory = f"/home/../multiTS/NFT/results/{data}/gan"
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{model}_results.xlsx")
    df = pd.DataFrame([kwargs])
    df.to_excel(path, index=False)
    

def add_gan_results_to_excel(model, data, **kwargs):
    directory = f"/home/../multiTS/NFT/results/{data}/gan"
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, f"{model}_results.xlsx")

    new_row_df = pd.DataFrame([kwargs])

    if os.path.exists(path):
        existing_df = pd.read_excel(path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        updated_df.to_excel(path, index=False)
    else:
        new_row_df.to_excel(path, index=False)


def get_original_loss(data, horizon):
    file = f'{BASE_PATH}results/{data}/nft/nft_{data}_results.xlsx'
    df = pd.read_excel(file)

    filtered_df = df[df['Horizon'] == horizon]
    test_mse, test_mae =None, None
    if not filtered_df.empty:
        last_row = filtered_df.iloc[-1]
        
        test_mse = last_row['test_mse']
        test_mae = last_row['test_mae']
    return test_mse, test_mae 