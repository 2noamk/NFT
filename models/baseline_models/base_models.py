import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, horizon):
        super(FC, self).__init__()
        self.horizon = horizon
        self.fc1 = nn.Linear(input_dim, output_dim)  # Input layer
        self.fc2 = nn.Linear(output_dim, output_dim)  # Hidden layer
        self.fc3 = nn.Linear(output_dim, output_dim)  # Hidden layer
        self.fc4 = nn.Linear(output_dim, output_dim)  # Hidden layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x
   
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


# class LSTM(nn.Module):
#     def __init__(self, 
#                  num_vars, 
#                  lookback, 
#                  horizon, 
#                  hidden_dim, 
#                  num_layers
#                  ):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim

#         self.lstm = nn.LSTM(num_vars, hidden_dim, num_layers, batch_first=True) #shape torch.Size([batch, lookback, hidden])

#         self.linear1 = nn.Linear(hidden_dim, num_vars) #shape torch.Size([batch, lookback, num_vars])
        
#         self.linear2 = nn.Linear(lookback, horizon) #shape torch.Size([batch, horizon, num_vars])

#     def forward(self, x):
#         device = x.device  # Get the device of the input data

#         # Initialize hidden and cell states and move them to the correct device
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()

#         # print("x", x.shape)
#         # print("h0", h0.shape)
#         # print("c0", c0.shape)

#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#         # print("out", out.shape)
        
#         out = self.linear1(out)
#         # print("out", out.shape)
        
#         out = self.linear2(out.transpose(2,1)).transpose(2,1)
#         # print("out", out.shape)
        
#         return out
   
#     def predict(self, x):
#         """
#         Predict the output given input data x.
#         :param x: Input data
#         :return: Model's predictions
#         """
#         self.eval()  # Set the model to evaluation mode
#         with torch.no_grad():
#             predictions = self.forward(x)
#         return predictions


class LSTM(nn.Module):
    def __init__(self, 
                 num_vars, 
                 lookback, 
                 horizon, 
                 hidden_dim, 
                 num_layers, 
                 dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(num_vars, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, num_vars)
        self.linear2 = nn.Linear(lookback, horizon)
        self.activation = nn.ReLU()

    def forward(self, x):
        device = x.device

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # [B, T, H]

        out = self.dropout(out)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out.transpose(2, 1)).transpose(2, 1)

        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class CNNBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 lookback, 
                 kernel_size, 
                 stride=1, 
                 padding=1, 
                 dropout=0.2):
        super(CNNBlock, self).__init__()
        def out_dim(input_size):
            return int(((input_size + padding - 1) / stride) + 1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.dropout1, nn.ReLU(),
            self.conv2, self.dropout2, nn.ReLU()
        )
        
        out2_dim = out_dim(out_dim(lookback))
        self.adjust_dim = nn.Linear(lookback-2, lookback)

    def forward(self, x):
        # print("Input shape:", x.shape) #Input shape: torch.Size([32, 5, 15])
        out1 = self.conv1(x)
        # print("After conv1:", out1.shape) # After conv1: torch.Size([32, 25, 16])
        
        out1 = self.dropout1(out1)
        out1 = nn.ReLU()(out1)

        out2 = self.conv2(out1)
        # print("After conv2:", out2.shape) # After conv2: torch.Size([32, 25, 17])
        
        out2 = self.dropout2(out2)
        # out2 = nn.ReLU()(out2)
        
        output = self.adjust_dim(out2)
        # print("After adjust_dim:", out2.shape)
        
        # output = self.adjust_dim2(output.transpose(1, 2)).transpose(1, 2)
        # print("After adjust_dim:", output.shape)
        return output


class CNN(nn.Module):
    def __init__(self, 
                 lookback, 
                 horizon, 
                 num_vars,
                 num_channels=[25, 50], 
                 kernel_size=2, 
                 dropout=0.2):
        super(CNN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = lookback if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * 0
            layers += [CNNBlock(in_channels, 
                                out_channels, 
                                num_vars, 
                                kernel_size=kernel_size, 
                                padding=padding,
                                dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear1 = nn.Linear(num_channels[-1], horizon)

    def forward(self, x):
        # print("x", x.shape)
        y1 = self.network(x)
        # print("y1", y1.shape)
        z1 =  self.linear1(y1.transpose(1, 2)).transpose(1, 2)
        # print("z1", z1.shape)
        return z1
    
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 lookback, 
                 horizon, 
                 num_vars,
                 num_layers, 
                 num_heads, 
                 dim_feedforward, 
                 dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Linear(num_vars, dim_feedforward)
        self.transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.output_layer1 = nn.Linear(dim_feedforward, num_vars)
        self.output_layer2 = nn.Linear(lookback, horizon)

    def forward(self, src):
        # print(src.shape)
        # Embedding input
        src = self.embedding(src)
        # print(src.shape)
        # Transformer requires a batch first input to be converted to sequence first
        src = src.permute(1, 0, 2)
        # print(src.shape)

        # Pass through the transformer
        output = self.transformer(src, src)
        # print(output.shape)
        # Convert back to batch first
        output = output.permute(1, 0, 2)
        # print(output.shape)

        # Pass through the output layer
        output = self.output_layer1(output)
        # print(output.shape)
        output = self.output_layer2(output.transpose(2,1)).transpose(2,1)
        # print(output.shape)
        return output
   
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_vars, 
                 kernel_size, 
                 stride=1, 
                 dilation=1, 
                 padding=1, 
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        def out_dim(input_size):
            return int(((input_size + padding - 1) / stride) + 1)
        
        self.conv1 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, 
                               out_channels, 
                               kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.dropout1, nn.ReLU(),
            self.conv2, self.dropout2, nn.ReLU()
        )
        
        out2_dim = out_dim(out_dim(num_vars))
        self.adjust_dim = nn.Linear(out2_dim, num_vars)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.dropout1(out1)
        out1 = nn.ReLU()(out1)
        out2 = self.conv2(out1)
        out2 = self.dropout2(out2)
        out2 = nn.ReLU()(out2)
        res = x if self.downsample is None else self.downsample(x)
        out2 = self.adjust_dim(out2)
        output = self.relu(out2 + res)
        return output
    

class TCN(nn.Module):
    def __init__(self, 
                 lookback, 
                 horizon, 
                 num_vars, 
                 num_channels=[25, 50], 
                 kernel_size=2, 
                 dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = lookback if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers += [TemporalBlock(in_channels, 
                                     out_channels, 
                                     num_vars, 
                                     kernel_size=kernel_size, 
                                     dilation=dilation, 
                                     padding=padding, 
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear1 = nn.Linear(num_channels[-1], horizon)

    def forward(self, x):
        y1 = self.network(x)
        z1 =  self.linear1(y1.transpose(1, 2)).transpose(1, 2)
        return z1
    
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

  
class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(
            self,
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            forecast_length=5,
            backcast_length=10,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None
    ):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        print('| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.backcast_length, self.forecast_length,
                    self.nb_harmonics
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
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
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

    def fit(self, x_train, y_train, validation_data=None, epochs=10, batch_size=32):

        def split(arr, size):
            arrays = []
            while len(arr) > size:
                slice_ = arr[:size]
                arrays.append(slice_)
                arr = arr[size:]
            arrays.append(arr)
            return arrays

        for epoch in range(epochs):
            x_train_list = split(x_train, batch_size)
            y_train_list = split(y_train, batch_size)
            assert len(x_train_list) == len(y_train_list)
            shuffled_indices = list(range(len(x_train_list)))
            random.shuffle(shuffled_indices)
            self.train()
            train_loss = []
            timer = time()
            for batch_id in shuffled_indices:
                batch_x, batch_y = x_train_list[batch_id], y_train_list[batch_id]
                self._opt.zero_grad()
                _, forecast = self(batch_x.clone().detach())
                loss = self._loss(forecast, squeeze_last_dim(batch_y.clone().detach()))
                train_loss.append(loss.item())
                loss.backward()
                self._opt.step()
            elapsed_time = time() - timer
            train_loss = np.mean(train_loss)

            test_loss = '[undefined]'
            if validation_data is not None:
                x_test, y_test = validation_data
                self.eval()
                _, forecast = self(x_test.clone().detach())
                test_loss = self._loss(forecast, squeeze_last_dim(y_test.clone().detach())).item()

            num_samples = len(x_train_list)
            time_per_step = int(elapsed_time / num_samples * 1000)
            print(f'Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs}')
            print(f'{num_samples}/{num_samples} [==============================] - ')
            print(f'{int(elapsed_time)}s {time_per_step}ms/step - ')
            # print(f'loss: {train_loss:.4f} - val_loss: {test_loss:.4f}')
            
    def predict(self, x, return_backcast=False):
        self.eval()
        b, f = self(x.clone().detach())
        b, f = b.detach().cpu().numpy(), f.detach().cpu().numpy()
        if len(x.shape) == 3:
            b = np.expand_dims(b, axis=-1)
            f = np.expand_dims(f, axis=-1)
        if return_backcast:
            return b
        return f

    @staticmethod
    def name():
        return 'NBeatsPytorch'

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, backcast):
        self._intermediary_outputs = []
        backcast = squeeze_last_dim(backcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast.to(f.device) + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append({'value': f.detach().numpy(), 'layer': layer_name})
        return backcast, forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(thetas.device))


def trend_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T.to(thetas.device))


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.backcast_linspace = linear_space(backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(backcast_length, forecast_length, is_forecast=True)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, thetas_dim, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast

