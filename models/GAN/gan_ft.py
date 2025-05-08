import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
BASE_PATH = "/home/../multiTS/NFT/"
sys.path.append(BASE_PATH)
from models.NFT.NFT import NFT
from dicts import data_to_num_vars_dict, data_to_label_len, data_to_steps, data_to_num_nft_blocks, data_to_num_of_series, single_data_to_series_list, noaa_series_to_years
from models.training_functions import get_data
from models.GAN.gan_functions import TimeSeriesDataset, FourierNoise, Noiser, train_gan_forecaster, evaluate_forecaster, get_original_loss, add_gan_results_to_excel

data = 'illness'
num_epochs = 10
n_vars = data_to_num_vars_dict.get(data, 5)
nb_blocks_per_stack = data_to_num_nft_blocks[data]

lookback, horizon = 10, 1
noise_scale = 0.05
attack_type = 'fourier'
batch_size = 32
hidden_dim = 32

for attack_type in ['fourier', 'noiser']:
    for lookback, horizon in data_to_steps[data]:
        for noise_scale in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            for hidden_dim in [8, 16, 32, 64, 128]:
                fine_tune = True  # Set to False if training from scratch
                model_name = f'nft_{lookback}l_{horizon}h_10epochs_{nb_blocks_per_stack}blocks_tcn'
                pretrained_model_path = f"{BASE_PATH}models/trained_models/{data}/nft/{model_name}/{model_name}.pth"


                train_X, train_y, val_X, val_y, test_X, test_y = get_data(
                    data=data, lookback=lookback, horizon=horizon,
                    n_series=data_to_num_of_series.get(data, 5),
                    print_stats=False, series=single_data_to_series_list.get(data, None),
                    year=noaa_series_to_years.get(data, None)
                )

                train_dataset = TimeSeriesDataset(train_X, train_y)
                test_dataset = TimeSeriesDataset(test_X, test_y)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


                forecaster = NFT(
                    forecast_length=horizon + data_to_label_len[data],
                    backcast_length=lookback,
                    n_vars=n_vars,
                    nb_blocks_per_stack=nb_blocks_per_stack,
                )
                if attack_type == 'fourier':
                    noiser = FourierNoise(n_vars, hidden_dim)
                else:
                    noiser = Noiser(n_vars, hidden_dim)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                forecaster.to(device)
                noiser.to(device)


                if fine_tune:
                    print(f"Loading pretrained model from {pretrained_model_path}...")
                    forecaster.load_state_dict(torch.load(pretrained_model_path))

                    # Freeze all layers except last 5
                    for param in list(forecaster.parameters())[:-5]:
                        param.requires_grad = False


                optimizer_forecaster = optim.Adam(filter(lambda p: p.requires_grad, forecaster.parameters()), lr=0.001)
                optimizer_noiser = optim.Adam(noiser.parameters(), lr=0.001)
                criterion = nn.MSELoss()


                forecaster_trained, noiser_trained = train_gan_forecaster(
                    train_loader=train_dataloader, test_loader=test_dataloader,
                    forecaster=forecaster, noiser=noiser,
                    optimizer_forecaster=optimizer_forecaster, optimizer_noiser=optimizer_noiser,
                    criterion=criterion, noise_scale=noise_scale, num_epochs=num_epochs
                )

                # Save the fine-tuned model if applicable
                # if fine_tune:
                #     torch.save(forecaster_trained.state_dict(), fine_tuned_model_path)
                #     print(f"Fine-tuned model saved to {fine_tuned_model_path}")


                clean_test_mse, clean_test_mae = evaluate_forecaster(forecaster_trained, test_dataloader, noiser=None)
                noisy_test_mse, noisy_test_mae = evaluate_forecaster(forecaster_trained, test_dataloader, noise_scale=noise_scale, noiser=noiser_trained)

                print(f"Test MSE (Clean): {clean_test_mse:.4f}, MAE (Clean): {clean_test_mae:.4f}")
                print(f"Test MSE (Noisy): {noisy_test_mse:.4f}, MAE (Noisy): {noisy_test_mae:.4f}")

                origin_mse, origin_mae = get_original_loss(data=data, horizon=horizon)

                add_gan_results_to_excel(
                    model="gan_finetune" if fine_tune else "gan", attack_type=attack_type, origin_mse=origin_mse, origin_mae=origin_mae, hidden_dim=hidden_dim,
                    data=data, lookback=lookback, horizon=horizon, epochs=num_epochs, noise_scale=noise_scale,
                    test_mse=clean_test_mse, noisy_test_mse=noisy_test_mse, test_mae=clean_test_mae, noisy_test_mae=noisy_test_mae
                )
