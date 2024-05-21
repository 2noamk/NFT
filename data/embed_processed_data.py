from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
import os

class dataset(Dataset):
    def __init__(self, X, y, lookback, horizon, label_len=0, data_stamp=None):
        self.seq_len = lookback
        self.pred_len = horizon
        self.label_len = label_len
        self.X = []
        self.y = []
        self.add_date = True if data_stamp is not None else False
        if self.add_date:
            self.X_mark = []
            self.y_mark = []
            
        for index in range(0, len(X) - (self.seq_len + self.pred_len)):  
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x =X[s_begin:s_end]
            seq_y = y[r_begin:r_end]
            seq_x_mark = data_stamp[s_begin:s_end]
            seq_y_mark = data_stamp[r_begin:r_end]
            
            self.X.append(seq_x)
            self.y.append(seq_y)
            if self.add_date:
                self.X_mark.append(seq_x_mark)
                self.y_mark.append(seq_y_mark)

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)
        if self.add_date:
            self.X_mark = np.stack(self.X_mark)
            self.y_mark = np.stack(self.y_mark)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.add_date: return self.X[idx], self.y[idx], self.X_mark[idx], self.y_mark[idx]
        return self.X[idx], self.y[idx]


def embed_data(data='traffic', lookback=96, label_len=0, horizons=[1, 16, 32, 48], add_data_stamp=False):
    base_path = f'/home/noam.koren/multiTS/NFT/data/{data}/' 

    for horizon in horizons:
        for flag in ['train', 'val', 'test']:
            X = torch.tensor(pd.read_pickle(f'{base_path}{flag}_X.pkl'))
            y = torch.tensor(pd.read_pickle(f'{base_path}{flag}_y.pkl'))
            data_stamp = torch.tensor(pd.read_pickle(f'{base_path}{flag}_data_stamp.pkl'))
            
            d = dataset(
                X=X, 
                y=y, 
                lookback=lookback, 
                horizon=horizon, 
                label_len=label_len, 
                data_stamp=data_stamp
            )
            
            path = f'{base_path}{data}_{lookback}l_{horizon}h_{label_len}label/'
            if add_data_stamp: mark_path = f'{base_path}{data}_date_stamp_{lookback}l_{horizon}h_{label_len}label/'

            # Check if directories exist, create if not
            if not os.path.exists(path):
                os.makedirs(path)
            if add_data_stamp and not os.path.exists(mark_path):
                os.makedirs(mark_path)


            # Convert tensor to a NumPy array if it's on GPU, you might need to call .cpu() before .numpy()
            X_np = d.X.cpu().numpy() if d.X.is_cuda else d.X.numpy()
            y_np = d.y.cpu().numpy() if d.y.is_cuda else d.y.numpy()

            # Save using pandas.to_pickle
            pd.to_pickle(X_np, f'{path}/{flag}_X.pkl')
            pd.to_pickle(y_np, f'{path}/{flag}_y.pkl')

            # For the date_stamp tensor, do the same if it's a tensor
            if add_data_stamp:
                if isinstance(d.X_mark, torch.Tensor):
                    X_mark_np = d.X_mark.cpu().numpy() if d.X_mark.is_cuda else d.X_mark.numpy()
                    pd.to_pickle(X_mark_np, f'{mark_path}/{flag}_X.pkl')
                    y_mark_np = d.y_mark.cpu().numpy() if d.y_mark.is_cuda else d.y_mark.numpy()
                    pd.to_pickle(y_mark_np, f'{mark_path}/{flag}_y.pkl')
                else:
                    pd.to_pickle(d.X_mark, f'{mark_path}/{flag}_X.pkl')
                    pd.to_pickle(d.y_mark, f'{mark_path}/{flag}_y.pkl')


if __name__ == "__main__":
    for data in ['exchange']:
        embed_data(
            data=data, 
            lookback=96, 
            label_len=0, 
            horizons=[720],
            add_data_stamp=False)      
