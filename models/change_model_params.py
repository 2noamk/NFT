import torch
import sys
import os


sys.path.append('NFT/')
from dicts import data_to_num_vars_dict, data_to_steps

sys.path.append('NFT/models/NFT/')
from NFT import NFT

sys.path.append('NFT/models/baseline_models/')
from base_models import TCN, TimeSeriesTransformer, LSTM

base_path = "NFT/models/trained_models/"

def get_model_object(model_type, lookback, horizon, num_vars):
    if model_type == 'nft':
        model = NFT(
            stack_types=('trend', 'seasonality'),
            nb_blocks_per_stack=2,
            forecast_length=horizon,
            backcast_length=lookback,
            n_vars=num_vars,
            layers_type='tcn',
            num_channels_for_tcn=[2, 2],
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None
        )
    elif model_type == 'tcn':
        model = TCN(
            lookback=lookback, 
            horizon=horizon, 
            num_vars=num_vars, 
            num_channels=[2, 2], 
            kernel_size=2, 
            dropout=0.2
        )
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(
            lookback=lookback, 
            horizon=horizon, 
            num_vars=num_vars,
            num_layers=1,
            num_heads=1,
            dim_feedforward=16,
        )
    elif model_type == 'lstm':
        model = LSTM(
            num_vars=num_vars, 
            lookback=lookback, 
            horizon=horizon, 
            hidden_dim=50, 
            num_layers=2
        )
    return model

def change_params(data, model_type, blocks):
    model_type = model_type
    data = data
    num_vars = data_to_num_vars_dict[data]

    for step in data_to_steps[data]:
        for epoch in [10, 20, 30, 40]:
            lookback, horizon = step
            
            model_name = f"{model_type}_{lookback}l_{horizon}h_{epoch}epochs_{blocks}blocks_tcn"
            model_path = f'{base_path}trained_models/{data}/{model_type}/{model_name}/{model_name}.pth'
import torch
import sys
import os


sys.path.append('NFT/')
from dicts import data_to_num_vars_dict, data_to_steps

sys.path.append('NFT/models/NFT/')
from NFT import NFT

sys.path.append('NFT/models/baseline_models/')
from base_models import TCN, TimeSeriesTransformer, LSTM

base_path = "NFT/models/trained_models/"

def get_model_object(model_type, lookback, horizon, num_vars):
    if model_type == 'nft':
        model = NFT(
            stack_types=('trend', 'seasonality'),
            nb_blocks_per_stack=2,
            forecast_length=horizon,
            backcast_length=lookback,
            n_vars=num_vars,
            layers_type='tcn',
            num_channels_for_tcn=[2, 2],
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None
        )
    elif model_type == 'tcn':
        model = TCN(
            lookback=lookback, 
            horizon=horizon, 
            num_vars=num_vars, 
            num_channels=[2, 2], 
            kernel_size=2, 
            dropout=0.2
        )
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(
            lookback=lookback, 
            horizon=horizon, 
            num_vars=num_vars,
            num_layers=1,
            num_heads=1,
            dim_feedforward=16,
        )
    elif model_type == 'lstm':
        model = LSTM(
            num_vars=num_vars, 
            lookback=lookback, 
            horizon=horizon, 
            hidden_dim=50, 
            num_layers=2
        )
    return model


def change_params(data, model_type, blocks, series='E00001'):
    model_type = model_type
    data = data
    num_vars = data_to_num_vars_dict[data]
    
    # model_path = f'{base_path}{data}/nft/nft_100l_1h_20epochs_2blocks_tcn_E00001/nft_100l_1h_20epochs_2blocks_tcn_E00001.pth'
    # model_object =  get_model_object(model_type, 100, 1, num_vars)

    # state_dict = torch.load(model_path)

    # adjusted_state_dict = {}
    # for key, value in state_dict.items():
    #     adjusted_key = key.replace('model.', '')  
    #     adjusted_state_dict[adjusted_key] = value

    # model_object.load_state_dict(adjusted_state_dict)
    i, j = 0, 0        
    for step in data_to_steps[data]:
        for epoch in [10, 20, 30, 40, 50]:
            lookback, horizon = step
            
            print(f"lookback={lookback}, horizon={horizon}, epoch={epoch}, blocks={blocks}")
            
            model_name = f"{model_type}_{lookback}l_{horizon}h_{epoch}epochs_{blocks}blocks_tcn"
            if data in ['ecg_single', 'eeg_single']: model_name = f'{model_name}_{series}'
            model_path = f'{base_path}{data}/{model_type}/{model_name}/{model_name}.pth'
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                i+=1
                continue

            model_object =  get_model_object(model_type, lookback, horizon, num_vars)

            state_dict = torch.load(model_path)

            adjusted_state_dict = {}
            for key, value in state_dict.items():
                adjusted_key = key.replace('model.', '')  
                adjusted_state_dict[adjusted_key] = value

            model_object.load_state_dict(adjusted_state_dict)
            j+=1
            # torch.save(adjusted_state_dict, model_path)
    print(i, j)

def main():
    data = "air_quality"
    for model_type in ['nft']:
        for blocks in [1, 2, 3, 4]:
            change_params(data, model_type, blocks)

if __name__ == "__main__":
    main()