import os
import sys
import torch
import pickle
import random
import numpy as np

from time import time
from typing import Union
from torch import nn, optim
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
sys.path.append('/home/../multiTS/NFT/')
from models.training_functions import plot_predictions, plot_loss_over_epoch_graph
from models.baseline_models.base_models import TCN, LSTM


class NFT(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(
            self,
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=2,
            forecast_length=5,
            backcast_length=10,
            thetas_dim=(4, 8),
            layers_type='tcn',
            num_channels_for_tcn=[25, 50],
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None,
            n_vars=1,
            is_poly=False,
            dft_1d=False
    ):
        super(NFT, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.n_vars = n_vars
        self.thetas_dim = thetas_dim
        self.layers_type = layers_type
        self.num_channels_for_tcn = num_channels_for_tcn
        self.is_poly = is_poly
        self.dft_1d = dft_1d
        self.parameters = []
        print('| NFT')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
        self._single_series = False

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NFT.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    units=self.hidden_layer_units, 
                    thetas_dim=self.thetas_dim[stack_id], 
                    layers_type=self.layers_type, 
                    n_vars=self.n_vars,
                    backcast_length=self.backcast_length, 
                    forecast_length=self.forecast_length,
                    num_channels_for_tcn=self.num_channels_for_tcn,
                    nb_harmonics=self.nb_harmonics,
                    is_poly=self.is_poly,
                )
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == NFT.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NFT.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def compile(self, loss: str, optimizer: Union[str, Optimizer]):
        if loss == 'mae':
            loss_ = l1_loss
        elif loss == 'mse':
            loss_ = mse_loss
        elif loss == 'cross_entropy':
            loss_ = cross_entropy
        elif loss == 'binary_crossentropy':
            loss_ = binary_cross_entropy
        else:
            raise ValueError(f'Unknown loss name: {loss}.')
        # noinspection PyArgumentList
        if isinstance(optimizer, str):
            if optimizer == 'adam':
                opt_ = optim.Adam
            elif optimizer == 'sgd':
                opt_ = optim.SGD
            elif optimizer == 'rmsprop':
                opt_ = optim.RMSprop
            else:
                raise ValueError(f'Unknown opt name: {optimizer}.')
            opt_ = opt_(lr=1e-4, params=self.parameters())
        else:
            opt_ = optimizer
        self._opt = opt_
        self._loss = loss_

    def fit(self, x_train, y_train, validation_data=None, epochs=10, batch_size=16, plot_epoch=10, 
            path_to_save_model=None, path_to_save_loss_plots=None, path_to_save_prediction_plots=None):

        def split(arr, size):
            arrays = []
            while len(arr) > size:
                slice_ = arr[:size]
                arrays.append(slice_)
                arr = arr[size:]
            arrays.append(arr)
            return arrays

        x_train = x_train.reshape(-1, self.backcast_length, self.n_vars)
        y_train = y_train.reshape(-1, self.forecast_length, self.n_vars)

        if validation_data is not None:
            x_val = validation_data[0].reshape(-1, self.backcast_length, self.n_vars)
            y_val = validation_data[1].reshape(-1, self.forecast_length, self.n_vars)  

        if path_to_save_model is not None and not os.path.exists(path_to_save_model):
            os.makedirs(path_to_save_model)
            
        if path_to_save_loss_plots is not None and not os.path.exists(path_to_save_loss_plots):
            os.makedirs(path_to_save_loss_plots)  
            
        if path_to_save_prediction_plots is not None and not os.path.exists(path_to_save_prediction_plots):
            os.makedirs(path_to_save_prediction_plots)    

        
        train_losses, val_losses = [], []
        min_train_loss = min_val_loss = float('inf')
        min_train_loss_epoch = min_val_loss_epoch = 0
        
        for epoch in range(epochs):
            x_train_list = split(x_train, batch_size)
            y_train_list = split(y_train, batch_size)
            assert len(x_train_list) == len(y_train_list)
            shuffled_indices = list(range(len(x_train_list)))
            random.shuffle(shuffled_indices)
            self.train()
            timer = time()
            total_loss_sum = 0.0
            total_data_points = 0

            for batch_id in shuffled_indices:
                batch_x, batch_y = x_train_list[batch_id], y_train_list[batch_id]
                self._opt.zero_grad()
                forecast = self(batch_x.clone().detach())
                loss = self._loss(forecast, batch_y.clone().detach())
                total_loss_sum += loss.item() * len(batch_x)
                total_data_points += len(batch_x)
                loss.backward()
                self._opt.step()
            train_loss = total_loss_sum / total_data_points
            if train_loss < min_train_loss: 
                min_train_loss = train_loss
                min_train_loss_epoch = epoch
            if epoch != 0: train_losses.append(train_loss)
            elapsed_time = time() - timer
            print(f"Epoch {epoch+1}, Average loss per data point: {train_loss}")

            val_loss = float('inf')
            if validation_data is not None:
                self.eval()
                forecast_val = self(x_val.clone().detach())
                val_loss = self._loss(forecast_val, y_val.clone().detach()).item() #/ (self.forecast_length * self.n_vars)
                if val_loss < min_val_loss: 
                    min_val_loss = val_loss
                    min_val_loss_epoch = epoch
            if epoch != 0: val_losses.append(val_loss)

            num_samples = len(x_train_list)
            time_per_step = int(elapsed_time / num_samples * 1000)

            print(f'Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs}')
            print(f'{num_samples}/{num_samples} [==============================] - '
                  f'{int(elapsed_time)}s {time_per_step}ms/step - '
                  f'loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
            
            if (epoch % plot_epoch == 0 or epoch == epochs - 1) and epoch != 0:
                plot_predictions(
                    y=y_val, 
                    predictions=forecast_val.detach().cpu().numpy(),
                    horizon=self.forecast_length, 
                    n_vars=self.n_vars,
                    epoch=epoch, 
                    path_to_save_prediction_plots=path_to_save_prediction_plots
                    )
                plot_loss_over_epoch_graph(
                    epoch, 
                    train_losses, 
                    val_losses, 
                    lookback=self.backcast_length,
                    horizon=self.forecast_length,
                    path_to_save_loss_plots=path_to_save_loss_plots
                    )
                if path_to_save_model is not None:
                    save_path = path_to_save_model + f"model_checkpoint_epoch_{epoch}.pth"
                    torch.save(self.state_dict(), save_path)
                    print(f"Saved model checkpoint at epoch {epoch} to {save_path}", flush=True)
        
        
        print(f"The min train loss is {min_train_loss} at epoch {min_train_loss_epoch}\n"
              f"The min val loss is {min_val_loss} at epoch {min_val_loss_epoch}\n")    
        
    def predict(self, x, return_interpretable=False, return_thetas=False):
        x = x.reshape(-1, self.backcast_length, self.n_vars)
        self.eval()
        f = self(x.clone().detach())
        f = f.detach().cpu()
        
        if return_interpretable or return_thetas:
            with torch.no_grad():
                trend_forecast, trend_thetas, _ = self(x, output_type='trend', return_thetas=True)
                seasonality_forecast, _, seasonality_thetas = self(x, output_type='seasonality', return_thetas=True)
                print(f'x: {x.shape}, trend_forecast: {trend_forecast.shape}, trend_thetas: {trend_thetas.shape}')
                print(f'x: {x.shape}, seasonality_forecast: {seasonality_forecast.shape}, seasonality_thetas: {seasonality_thetas.shape}')

            if return_interpretable and return_thetas:
                return f, trend_forecast, seasonality_forecast, trend_thetas, seasonality_thetas
            elif return_thetas:
                return trend_thetas, seasonality_thetas
            else:
                return f, trend_forecast, seasonality_forecast

        else:
            return f

    @staticmethod
    def name():
        return 'NFTPytorch'

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, backcast, output_type='all', return_thetas=False):
        if backcast.dim() == 2:
            self._single_series = True
            backcast = backcast.unsqueeze(-1)
        self._intermediary_outputs = []
        batch_size = backcast.shape[0]
        forecast = torch.zeros(size=(batch_size, self.forecast_length, self.n_vars))
        trend_thetas_sum = torch.zeros(size=(batch_size, self.n_vars, self.thetas_dim[0]))
        seasonality_thetas_sum = torch.zeros(size=(batch_size, self.n_vars, self.thetas_dim[1]))

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                block = self.stacks[stack_id][block_id]
                block_type = block.__class__.__name__
                if (output_type == 'all' or
                    (output_type == 'trend' and block_type == 'TrendBlock') or
                    (output_type == 'seasonality' and block_type == 'SeasonalityBlock')):


                    if return_thetas:
                        if block_type == 'SeasonalityBlock': b, f, theta_f = block(backcast, return_thetas, self.dft_1d)
                        else: b, f, theta_f = block(backcast, return_thetas)
                    else: 
                        if block_type == 'SeasonalityBlock': b, f = block(backcast, return_thetas, self.dft_1d)
                        else: b, f = block(backcast, return_thetas)
                    
                    backcast = backcast - b  
                    forecast = forecast.to(f.device) + f

                    layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                    if self._gen_intermediate_outputs:
                        self._intermediary_outputs.append({'value': f.detach().cpu().numpy(), 'layer': layer_name})

                    if return_thetas: 
                        trend_thetas_sum = trend_thetas_sum.to(f.device)
                        seasonality_thetas_sum = seasonality_thetas_sum.to(f.device)
                        if block_type == 'TrendBlock': trend_thetas_sum += theta_f
                        elif block_type == 'SeasonalityBlock': seasonality_thetas_sum += theta_f

        if self._single_series:
            backcast, forecast = backcast.squeeze(), forecast.squeeze()

        if return_thetas:
            return forecast, trend_thetas_sum, seasonality_thetas_sum
        else:
            return forecast


def seasonality_model(thetas, len_series_linespace, n_vars_linespace, dft_1d=False):
    """
    thetas: is of size (batch_size, n_vars, thetas_dim)s
    len_series_linespace: linespace of length: series_len
    vars_linespace: linespace of length: n_vars

    Extention to multivariate: 
    2 Fourier matrix: M1 - mat of shape (n_vars, n_vars) 
                      M2 - mast of shape (thetas_dim, series_len)
    Then we will multiply: M1 * thetas * M2 
    """

    def create_fourier_mat(p, l):
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * l) for i in range(p1)])).float() # H/2-1
        s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * l) for i in range(p2)])).float()
        return torch.cat([s1, s2]) 

    batch_size, n_vars, thetas_dim = thetas.shape

    M1 = create_fourier_mat(n_vars, n_vars_linespace).unsqueeze(0).expand(batch_size, -1, -1)
    M2 = create_fourier_mat(thetas_dim, len_series_linespace).unsqueeze(0).expand(batch_size, -1, -1)

    if thetas.is_cuda:
        M1 = M1.to(thetas.device)
        M2 = M2.to(thetas.device)

    intermediate = torch.bmm(thetas, M2) # shape: (batch_size, n_vars, series_len)
    if dft_1d:
        print('usinf 1ddft') 
        return intermediate.permute(0, 2, 1)  # shape: (batch_size, series_len, n_vars)
    
    # print('usinf 2ddft') 
    result = torch.bmm(M1, intermediate) # shape: (batch_size, n_vars, series_len)
    result = result.permute(0, 2, 1)  # shape: (batch_size, series_len, n_vars)
    return result


def trend_model(thetas, t, n=None, is_poly=False):
    """
    thetas: is of size (batch_size, n_vars, thetas_dim)
    t: linspace of length: series_len
    n: linspace of length: num of vars
    """
    def create_trend_mat(h, d):
        return torch.tensor(np.array([h ** i for i in range(d)])).float().to(thetas.device)

    batch_size, n_vars, p = thetas.shape
    # assert p <= 4, 'thetas_dim is too big.'
    
    T2 = create_trend_mat(t, p).unsqueeze(0).expand(batch_size, -1, -1)
    
    if is_poly:
        T1 = create_trend_mat(n, n_vars).unsqueeze(0).expand(batch_size, -1, -1) # instaed of p, n_vars
        thetas = torch.bmm(T1, thetas) # shape: (batch_size, n_vars, thetas_dim)

    trends = torch.bmm(thetas, T2) # shape: (batch_size, n_vars, series_len)
    trends = trends.permute(0, 2, 1)  # shape: (batch_size, series_len, n_vars)
    return trends


def linear_space(horizon):
    return np.arange(0, horizon) / horizon


class Block(nn.Module):

    def __init__(self, units, thetas_dim, layers_type, n_vars, backcast_length=10, forecast_length=5, num_channels_for_tcn=[25, 50], share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.layers_type = layers_type
        self.n_vars = n_vars
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        if self.layers_type == 'tcn':
            self.tcn = TCN(
                lookback=backcast_length, 
                horizon=units, 
                num_vars=n_vars,
                num_channels=num_channels_for_tcn, 
                kernel_size=2, 
                dropout=0.2)
        
        elif self.layers_type == 'fc':
            self.fc1 = nn.Linear(backcast_length * n_vars, units,)
            self.fc2 = nn.Linear(units, units)
            self.fc3 = nn.Linear(units, units)
            self.fc4 = nn.Linear(units, units)
            
        elif self.layers_type == 'lstm':
            self.lstm = LSTM(
                    num_vars=n_vars, 
                    lookback=backcast_length, 
                    horizon=units, 
                    hidden_dim=50, 
                    num_layers=2
                    )

        self.backcast_linspace = linear_space(backcast_length)
        self.forecast_linspace = linear_space(forecast_length)
        self.vars_linespace = linear_space(n_vars)
        if share_thetas:
            if self.layers_type in ['tcn', 'lstm']:
                self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)

            if self.layers_type == 'fc':
                self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim * n_vars, bias=False)
                
        else:
            if self.layers_type in ['tcn', 'lstm']:
                self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
                self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

            if self.layers_type == 'fc':
                self.theta_b_fc = nn.Linear(units, thetas_dim * n_vars, bias=False)
                self.theta_f_fc = nn.Linear(units, thetas_dim * n_vars, bias=False)

    def forward(self, x):
        """
        x is of size (batch_size, series_len, n_vars)
        """
        if self.layers_type == 'tcn':
            x = self.tcn(x).transpose(1, 2) # x is of shape: (batch, n_vars, units)
        
        if self.layers_type == 'lstm':
            x = self.lstm(x).transpose(1, 2) # x is of shape: (batch, n_vars, units)
        
        if self.layers_type == 'fc':
            batch_size, series_len, n_vars = x.shape
            x = x.reshape(batch_size, series_len * n_vars)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))  # x is of shape: (batch, units)
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, layers_type, n_vars, backcast_length=10, forecast_length=5, num_channels_for_tcn=[25, 50], is_poly=False, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, layers_type, n_vars, backcast_length,
                                                   forecast_length, num_channels_for_tcn, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, thetas_dim, layers_type, n_vars, backcast_length,
                                                   forecast_length, num_channels_for_tcn, share_thetas=True)

    def forward(self, x, return_thetas=False, dft_1d=False):
        """
        x is of size (batch_size, series_len, n_vars)
        return:
        backcast: is of size (batch_size, backcast_len, n_vars)
        forecast: is of size (batch_size, forecast_len, n_vars)
        """
        x = super(SeasonalityBlock, self).forward(x)

        if self.layers_type in ['tcn', 'lstm']:
            backcast_thetas = self.theta_b_fc(x)
            forecast_thetas = self.theta_f_fc(x)
             
        if self.layers_type == 'fc':
            batch_size = x.shape[0]
            backcast_thetas = self.theta_b_fc(x).reshape(batch_size, self.n_vars, self.thetas_dim) 
            forecast_thetas = self.theta_f_fc(x).reshape(batch_size, self.n_vars, self.thetas_dim)

        backcast = seasonality_model(backcast_thetas, self.backcast_linspace, self.vars_linespace, dft_1d)
        forecast = seasonality_model(forecast_thetas, self.forecast_linspace, self.vars_linespace, dft_1d)

        if return_thetas: return backcast, forecast, forecast_thetas
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, layers_type, n_vars, backcast_length=10, forecast_length=5, num_channels_for_tcn=[25, 50], 
                 nb_harmonics=None, is_poly=False):
        super(TrendBlock, self).__init__(units, thetas_dim, layers_type, n_vars, backcast_length,
                                         forecast_length, num_channels_for_tcn, share_thetas=True)
        self.is_poly = is_poly
        self.vars_linspace = linear_space(n_vars) if is_poly else None
            
    def forward(self, x, return_thetas=False):
        """
        x: is of size (batch_size, series_len, n_vars)
        return:
        backcast: is of size (batch_size, backcast_len, n_vars)
        forecast: is of size (batch_size, forecast_len, n_vars)
        """
        x = super(TrendBlock, self).forward(x) 
        
        batch_size = x.shape[0]
        backcast_thetas = self.theta_b_fc(x).reshape(batch_size, self.n_vars, self.thetas_dim) 
        forecast_thetas = self.theta_f_fc(x).reshape(batch_size, self.n_vars, self.thetas_dim)
        
        backcast = trend_model(backcast_thetas, self.backcast_linspace, self.vars_linspace, self.is_poly)
        forecast = trend_model(forecast_thetas, self.forecast_linspace, self.vars_linspace, self.is_poly)

        if return_thetas: return backcast, forecast, forecast_thetas
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, layers_type, n_vars, backcast_length=10, forecast_length=5, num_channels_for_tcn=[25, 50], 
                 nb_harmonics=None, is_poly=False):
        super(GenericBlock, self).__init__(units, thetas_dim, 'fc', n_vars, backcast_length, 
                                           forecast_length, num_channels_for_tcn, share_thetas=True)

        self.backcast_fc = nn.Linear(thetas_dim * n_vars, backcast_length * n_vars)
        self.forecast_fc = nn.Linear(thetas_dim * n_vars, forecast_length * n_vars)

    def forward(self, x, return_thetas=False):
        """
        x is of size (batch_size, series_len, n_vars)
        return:
        backcast: is of size (batch_size, backcast_len, n_vars)
        forecast: is of size (batch_size, forecast_len, n_vars)
        """
        x = super(GenericBlock, self).forward(x) # x is of size (batch_size, units)
        
        batch_size = x.shape[0]
        theta_b = self.theta_b_fc(x) # theta_b is of size (batch_size, thetas_dim * n_vars)
        theta_f = self.theta_f_fc(x) # theta_f is of size (batch_size, thetas_dim * n_vars)

        backcast = self.backcast_fc(theta_b).reshape(batch_size, self.backcast_length, self.n_vars) 
        forecast = self.forecast_fc(theta_f).reshape(batch_size, self.forecast_length, self.n_vars) 
        
        return backcast, forecast
    
    
    
    
    
    
    
    import os
