import sys
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from nbeats_pytorch.model import NBeatsNet
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/home/noam.koren/multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_num_of_series, data_to_steps, single_data_to_series_list
from models.training_functions import get_data, calculate_smape, calculate_mape, calculate_mase, add_results_to_excel, get_model_name, get_path, save_model
from models.NFT.NFT import NFT
from models.baseline_models.base_models import NBeatsNet, TCN, TimeSeriesTransformer, LSTM

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
        num_channels=[25, 50], 
        kernel_size=2, 
        dropout=0.2,
        data='noaa', 
        print_stats=False,
        n_series=36, 
        series="AG000060590",
        idx=None
        ):
        super(Model, self).__init__()
        self.model= NBeatsNet(
            stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
            forecast_length=horizon,
            backcast_length=lookback,
            nb_blocks_per_stack=nb_blocks_per_stack,
            thetas_dim=(4,8),
            )
        self.data = data 
        self.lookback=lookback 
        self.horizon=horizon 
        self.num_vars = num_vars
        self.n_series = n_series
        self.series = series
        self.print_stats = print_stats
        self.idx = idx
        
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
        print("training_step")
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
        train_y = train_y.reshape(-1, self.horizon, self.num_vars)
        
        val_X = val_X.reshape(-1, self.lookback, self.num_vars)
        val_y = val_y.reshape(-1, self.horizon, self.num_vars)
        
        test_X = test_X.reshape(-1, self.lookback, self.num_vars)
        test_y = test_y.reshape(-1, self.horizon, self.num_vars)
        
        print(f"train_X.shape = {train_X.shape}")

        if self.idx is not None:
            train_X, train_y = train_X[:,:,self.idx], train_y[:,:,self.idx]
            val_X, val_y = val_X[:,:,self.idx], val_y[:,:,self.idx]
            test_X, test_y = test_X[:,:,self.idx], test_y[:,:,self.idx]
        

        print(f"train_X.shape = {train_X.shape}")
        
        self.train_dataset = TensorDataset(train_X, train_y)
        self.val_dataset = TensorDataset(val_X, val_y)
        self.test_dataset = TensorDataset(test_X, test_y)
        print("Data prepered")
        
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
        print("on_train_epoch_end")
        super().on_train_epoch_end()
        if self.batch_count > 0:
            avg_loss = self.cumulative_loss / self.batch_count
            self.train_loss.append(avg_loss)
            
            # Reset accumulators for the next epoch
            self.cumulative_loss = 0.0
            self.batch_count = 0
        print("on_train_epoch_end2")
            
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
            
    def compute_metrics(self, dataloader):
        smape_values, mape_values, mase_values = [], [], []
        for batch in dataloader:
            x, y = batch
            y_hat = self.model.predict(x)
            smape_values.append(calculate_smape(y, y_hat).item())
            mape_values.append(calculate_mape(y, y_hat).item())
            mase_result = calculate_mase(y, y_hat)
            if torch.is_tensor(mase_result):
                mase_values.append(mase_result.item())
            else:
                mase_values.append(mase_result)
        return {
            'smape': np.mean(smape_values),
            'mape': np.mean(mape_values),
            'mase': np.mean(mase_values)
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
    num_channels=None
): 
    print(f"training model {model_type} on data {data} for lookback {lookback} and horizon {horizon}")
    
    num_of_vars = data_to_num_vars_dict[data]    
    n_series = data_to_num_of_series[data]
    
    train_metrics_idx, val_metrics_idx, test_metrics_idx = [], [], []
    
    for idx in range(num_of_vars):
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
            idx=idx,
            )

        trainer = pl.Trainer(
            max_epochs=num_epochs, 
            accelerator='cuda', 
            devices=torch.cuda.device_count(),
            log_every_n_steps=32
        )

        trainer.fit(model)
        
        trainer.test(model)
            
        train_metrics_idx.append(model.compute_metrics(model.train_dataloader()))
        val_metrics_idx.append(model.compute_metrics(model.val_dataloader()))
        test_metrics_idx.append(model.compute_metrics(model.test_dataloader()))
        
        print(f"test metrics = {train_metrics_idx[-1]}")

    def aggregate_metrics(metrics_list):
        aggregated_metrics = {}
        for metric in metrics_list[0].keys():
            aggregated_metrics[metric] = np.mean([m[metric] for m in metrics_list])
        return aggregated_metrics

    train_metrics = aggregate_metrics(train_metrics_idx)
    val_metrics = aggregate_metrics(val_metrics_idx)
    test_metrics = aggregate_metrics(test_metrics_idx)

    print("Aggregated Train Metrics:", train_metrics)
    print("Aggregated Validation Metrics:", val_metrics)
    print("Aggregated Test Metrics:", test_metrics)

    add_results_to_excel(
        model=model_type, 
        data=data, 
        lookback=lookback, 
        horizon=horizon, 
        epochs=num_epochs, 
        blocks=blocks, 
        series=series, 
        train_mse=model.train_loss[-1], 
        test_mse=model.test_loss[-1], 
        train_smape=train_metrics['smape'], 
        test_smape=test_metrics['smape'], 
        train_mape=train_metrics['mape'], 
        test_mape=test_metrics['mape'], 
        train_mase=train_metrics['mase'], 
        test_mase=test_metrics['mase']
        )
    
    return


def main():
    model_type = 'nbeats'
    data = 'chorales'
    out_txt_name = f"{model_type}_{data}.txt"
    num_channels = [2, 2]
    epochs = [10]
    blocks = 2
    layers_type = 'tcn'
    print(f"data = {data}")
    
    if data in ['eeg_single', 'ecg_single', 'noaa']:
        for series in single_data_to_series_list[data]:
            for s in data_to_steps[data]:
                for num_epochs in epochs:
                    lookback, horizon = s
                    train_lightning_model(
                        model_type=model_type,
                        data=data,
                        lookback=lookback,
                        horizon=horizon,
                        num_epochs=num_epochs,
                        blocks=blocks,
                        out_txt_name=out_txt_name,
                        print_stats=False,
                        series=series,
                        layers_type=layers_type,
                        num_channels=num_channels,
                    )
    else:
        for s in data_to_steps[data]:
            for num_epochs in epochs:
                lookback, horizon = s
                train_lightning_model(
                    model_type=model_type,
                    data=data,
                    lookback=lookback,
                    horizon=horizon,
                    num_epochs=num_epochs,
                    blocks=blocks,
                    out_txt_name=out_txt_name,
                    print_stats=False,
                    series=None,
                    layers_type=layers_type,
                    num_channels=num_channels,
                )


if __name__ == "__main__":
    main()