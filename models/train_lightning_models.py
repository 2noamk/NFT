import sys
import torch
import time
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from nbeats_pytorch.model import NBeatsNet
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/home/../multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_num_nft_blocks, data_to_num_of_series, data_to_steps, single_data_to_series_list, data_to_label_len
# from lists import noaa_years
from models.training_functions import add_run_time_to_excel, calculate_rmsse, add_num_of_params_to_excel, get_data, calculate_smape, calculate_mape, calculate_mase, calculate_mae, add_results_to_excel, get_model_name, get_path, save_model
from models.NFT.NFT import NFT
from models.baseline_models.base_models import TCN, TimeSeriesTransformer, LSTM
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')


class Model(pl.LightningModule):
    def __init__(self,
        model='nft',
        lookback=15, 
        horizon=1, 
        num_vars=1,
        nb_blocks_per_stack=2,
        thetas_dim=(4, 8),
        layers_type='tcn',
        num_channels=5, 
        kernel_size=2, 
        dropout=0.2,
        data='noaa', 
        print_stats=False,
        n_series=36, 
        series="AG000060590",
        is_poly=False,
        stack_types=('trend', 'seasonality'),
        dft_1d=False
        ):
        super(Model, self).__init__()
        
        self.model_name = model
        self.data = data 
        self.lookback=lookback 
        self.horizon=horizon
        self.label_len = 0# data_to_label_len[data]
        self.num_vars = num_vars
        self.n_series = n_series
        self.series = series
        self.print_stats = print_stats
        

        if self.model_name == 'nft':
            self.model = NFT(
                stack_types=stack_types,
                nb_blocks_per_stack=nb_blocks_per_stack,
                forecast_length=horizon,
                backcast_length=lookback,
                n_vars=num_vars,
                thetas_dim=thetas_dim,
                layers_type=layers_type,
                num_channels_for_tcn=num_channels,
                share_weights_in_stack=False,
                hidden_layer_units=256,
                nb_harmonics=None,
                is_poly=is_poly,
                dft_1d=dft_1d
            )
            
        elif self.model_name == 'tcn':
            self.model = TCN(
            lookback=lookback, 
            horizon=horizon + self.label_len, 
            num_vars=num_vars, 
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            dropout=dropout
            )
        elif self.model_name == 'transformer':
            self.model = TimeSeriesTransformer(
            lookback=lookback, 
            horizon=horizon + self.label_len, 
            num_vars=num_vars,
            num_layers=1,
            num_heads=1,
            dim_feedforward=16,
            )
        elif self.model_name == 'lstm':
            self.model = LSTM(
                num_vars=num_vars, 
                lookback=lookback, 
                horizon=horizon + self.label_len, 
                hidden_dim=50, 
                num_layers=2
            )
        elif self.model_name == 'nbeats':
            self.model = NBeatsNet(
                stack_types=(NBeatsNet.TREND_BLOCK_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
                nb_blocks_per_stack=2,
                thetas_dim=(4,8),
                forecast_length=horizon + self.label_len,
                backcast_length=lookback,
                )
        
        
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        
        self.cumulative_loss = 0.0
        self.batch_count = 0
        self.cumulative_val_loss = 0.0
        self.val_batch_count = 0
        self.cumulative_test_loss = 0.0
        self.test_batch_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        self.cumulative_loss += loss.item()
        self.batch_count += 1

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)

        # Update accumulators
        self.cumulative_val_loss += loss.item()
        self.val_batch_count += 1

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
   
        # Update accumulators
        self.cumulative_test_loss += loss.item()
        self.test_batch_count += 1

        return {'test_loss': loss}
    
    def configure_optimizers(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total trainable parameters:", total_params)
        # add_num_of_params_to_excel(dataset_name=self.data, model=self.model_name, total_params=total_params)
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def prepare_data(self):
        train_X, train_y, val_X, val_y, test_X, test_y = get_data(
        data=self.data, 
        lookback=self.lookback, 
        horizon=self.horizon,
        n_series=self.n_series,
        series=self.series,
        print_stats=self.print_stats
        )
        
        train_X = train_X.reshape(-1, self.lookback, self.num_vars)
        train_y = train_y.reshape(-1, self.horizon + self.label_len, self.num_vars)
        
        val_X = val_X.reshape(-1, self.lookback, self.num_vars)
        val_y = val_y.reshape(-1, self.horizon + self.label_len, self.num_vars)
        
        test_X = test_X.reshape(-1, self.lookback, self.num_vars)
        test_y = test_y.reshape(-1, self.horizon + self.label_len, self.num_vars)
        
        self.train_dataset = TensorDataset(train_X, train_y)
        self.val_dataset = TensorDataset(val_X, val_y)
        self.test_dataset = TensorDataset(test_X, test_y)
                
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=4
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=32,
            num_workers=4
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=32,
            num_workers=4
            )

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.batch_count > 0:
            avg_loss = self.cumulative_loss / self.batch_count
            self.train_loss.append(avg_loss)
            
            # Reset accumulators for the next epoch
            self.cumulative_loss = 0.0
            self.batch_count = 0
            
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.val_batch_count > 0:
            avg_loss = self.cumulative_val_loss / self.val_batch_count
            self.val_loss.append(avg_loss)

            self.cumulative_val_loss = 0.0
            self.val_batch_count = 0

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.test_batch_count > 0:
            avg_loss = self.cumulative_test_loss / self.test_batch_count
            self.test_loss.append(avg_loss)

            # Reset accumulators for the next epoch
            self.cumulative_test_loss = 0.0
            self.test_batch_count = 0
 
    def calculate_mase(y_true, y_pred, seasonal_period=1):
        # Naive forecast (shifted true values by the seasonal period)
        naive_forecast = torch.roll(y_true, shifts=seasonal_period, dims=0)
        naive_errors = torch.mean(torch.abs(y_true[seasonal_period:] - naive_forecast[seasonal_period:]))
        prediction_errors = torch.mean(torch.abs(y_true - y_pred))
        return prediction_errors / naive_errors

    def compute_metrics(self, dataloader):
        preds, trues = [], []

        for batch in dataloader:
            x, y = batch
            y_hat = self.model.predict(x)
            preds.append(y_hat.detach().cpu())
            trues.append(y.detach().cpu())

        # Concatenate all batch predictions and true values
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        # Calculate metrics on the entire dataset
        mae = torch.mean(torch.abs(preds - trues)).item()
        mse = torch.mean((preds - trues) ** 2).item()
        rmse = torch.sqrt(torch.mean((preds - trues) ** 2)).item()
        mspe = torch.mean(((preds - trues) / trues) ** 2).item()
        mape = calculate_mape(trues, preds).item()
        smape = calculate_smape(trues, preds).item()
        mase = calculate_mase(trues, preds).item()
        rmsse = calculate_rmsse(trues, preds).item()

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mspe': mspe,
            'mape': mape,
            'smape': smape,
            'mase': mase,
            'rmsse': rmsse,
        }


def train_lightning_model(
    model_type,
    data, 
    lookback, 
    horizon, 
    num_epochs, 
    blocks, 
    out_txt_name,
    print_stats, 
    series,
    thetas_dim=(4,8),
    layers_type='tcn',
    num_channels=None,
    is_poly=False,
    year=None,
    stack_types=('trend', 'seasonality'),
    dft_1d=False
    ): 
    print(f"training model {model_type} on data {data} for lookback {lookback} and horizon {horizon}")
    

    num_of_vars=data_to_num_vars_dict.get(data, 5) 
    n_series=data_to_num_of_series.get(data, 1)
    
    if year is not None:
        data = f'noaa_{series}_{year}'
    
    model = Model(
        model=model_type,
        lookback=lookback, 
        horizon=horizon, 
        num_vars=num_of_vars,
        nb_blocks_per_stack=blocks,
        num_channels=num_channels, 
        thetas_dim=thetas_dim,
        layers_type=layers_type,
        kernel_size=2, 
        dropout=0.2,
        data=data, 
        print_stats=print_stats,
        n_series=n_series, 
        series=series,
        is_poly=is_poly,
        stack_types=stack_types,
        dft_1d=dft_1d
        )

    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        log_every_n_steps=32
    )

    # start_time = time.time()

    # Fit model
    trainer.fit(model)

    # end_time = time.time()
    
    # add_run_time_to_excel(dataset_name=data, 
    #                       model=model_type, 
    #                       time=end_time-start_time)
    
    trainer.test(model)
          
    train_metrics = model.compute_metrics(model.train_dataloader())
    val_metrics = model.compute_metrics(model.val_dataloader())
    test_metrics = model.compute_metrics(model.test_dataloader())

    add_results_to_excel(
        model=model_type, 
        data=data, 
        lookback=lookback, 
        horizon=horizon, 
        epochs=num_epochs, 
        blocks=blocks, 
        stack_types=stack_types,
        num_channels=num_channels,
        series=series, 
        train_mse=train_metrics['mse'], 
        test_mse=test_metrics['mse'], 
        train_mae=train_metrics['mae'], 
        test_mae=test_metrics['mae'], 
        train_smape=train_metrics['smape'], 
        test_smape=test_metrics['smape'], 
        train_mape=train_metrics['mape'], 
        test_mape=test_metrics['mape'], 
        train_mase=train_metrics['mase'], 
        test_mase=test_metrics['mase'],
        train_rmsse=train_metrics['rmsse'], 
        test_rmsse=test_metrics['rmsse'],
        )
   
    # model_name = get_model_name(
    #     model_type=model_type, 
    #     data=data, 
    #     lookback=lookback, 
    #     horizon=horizon, 
    #     epochs=num_epochs,
    #     blocks=blocks,  
    #     layers_type='tcn', 
    #     n_series=n_series,
    #     series=series
    #     )
    
    # path_to_save_model, _, _, _ = get_path(
    #     model_name=model_name, 
    #     data=data,
    #     model_type=model_type
    # )
    
    # save_model(model, path_to_save_model, model_name)
    
    return


def main():
    """Choose model: nft / tcn / transformer / lstm / patchtst"""
    model_type = 'tcn'
    """Choose dataset: illness / air_quality / noaa / ecg / ecg_single / eeg_single / chorales"""
    data = 'noaa'
    out_txt_name = f"{model_type}_{data}.txt"
    tcn_channels = [[25, 50]]
    num_channels = [2,2]
    num_epochs = 10
    blocks = 2
    layers_type = 'tcn'
    is_poly = False
    thetas_dim = (4,8)
    stacks = ('trend', 'seasonality')
    dft_1d=False

    print(f"data = {data}")


    if data == 'seasonal_trend_test':
        for seasonal_amplitude in [0.75, 1.0]:
            for trend_amplitude in [0.75]:
                data = f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
                for lookback, horizon in data_to_steps['seasonal_trend_0_5']:     
                    train_lightning_model(
                        model_type=model_type,
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=10,
                        blocks=num_channels,
                        out_txt_name=out_txt_name,
                        print_stats=False,
                        series=None,
                        layers_type=layers_type,
                        num_channels=num_channels,
                        )
        for seasonal_amplitude in [0.25, 0.75]:
            for trend_amplitude in [0.5]:
                data = f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
                for lookback, horizon in data_to_steps['seasonal_trend_0_5']:     
                    train_lightning_model(
                        model_type=model_type,
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=10,
                        blocks=num_channels,
                        out_txt_name=out_txt_name,
                        print_stats=False,
                        series=None,
                        layers_type=layers_type,
                        num_channels=num_channels,
                        )
        for seasonal_amplitude in [0.25, 1.25, 1.75, 2.25, 2.5, 3.5]:
            for trend_amplitude in [0.25]:
                data = f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
                for lookback, horizon in data_to_steps['seasonal_trend_0_5']:     
                    train_lightning_model(
                        model_type=model_type,
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=10,
                        blocks=num_channels,
                        out_txt_name=out_txt_name,
                        print_stats=False,
                        series=None,
                        layers_type=layers_type,
                        num_channels=num_channels,
                        )
        for seasonal_amplitude in [6,7,9]:
            for trend_amplitude in [0.2]:
                data = f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
                for lookback, horizon in data_to_steps['seasonal_trend_0_5']:     
                    train_lightning_model(
                        model_type=model_type,
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=10,
                        blocks=num_channels,
                        out_txt_name=out_txt_name,
                        print_stats=False,
                        series=None,
                        layers_type=layers_type,
                        num_channels=num_channels,
                        )
    else:
        for s in data_to_steps[data]:
            lookback, horizon = s
            if data in ['eeg_single', 'ecg_single', 'noaa', 'ett']:
                for series in single_data_to_series_list[data]:
                        train_lightning_model(
                            model_type=model_type,
                            data=data,
                            lookback=lookback,
                            horizon=horizon,
                            num_epochs=num_epochs,
                            blocks=data_to_num_nft_blocks.get(data, 2),
                            out_txt_name=out_txt_name,
                            print_stats=False,
                            series=series,
                            thetas_dim=thetas_dim,
                            layers_type=layers_type,
                            num_channels=num_channels,
                            is_poly=is_poly
                    )
            elif data == 'noaa_years':
                for series in single_data_to_series_list[data[0:4]]:
                    for year in noaa_years:
                        train_lightning_model(
                            model_type=model_type,
                            data=data,
                            lookback=lookback,
                            horizon=horizon,
                            num_epochs=num_epochs,
                            blocks=data_to_num_nft_blocks.get(data, 2),
                            out_txt_name=out_txt_name,
                            print_stats=False,
                            series=series,
                            thetas_dim=thetas_dim,
                            layers_type=layers_type,
                            num_channels=num_channels,
                            is_poly=is_poly,
                            year=year,
                    )
            else: 
                train_lightning_model(
                        model_type=model_type,
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=num_epochs,
                        blocks=data_to_num_nft_blocks.get(data, 2),
                        out_txt_name=out_txt_name,
                        print_stats=True,
                        series=None,
                        thetas_dim=thetas_dim,
                        stack_types=stacks,
                        layers_type=layers_type,
                        num_channels=num_channels,
                        is_poly=is_poly,
                        dft_1d=dft_1d
                        )
    


if __name__ == "__main__":
    main()