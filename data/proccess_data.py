import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import sys

sys.path.append('/home/../multiTS/NFT/data/')
from augmentation import run_augmentation_single

warnings.filterwarnings('ignore')



def process_data(seq_len, root_path, dataset_name, flag='train',
                 features='S', data_path='ETTh1.csv', save_data=True,
                 target='OT', scale=True):
    assert flag in ['train', 'test', 'val']
    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]
    scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    print(f"{flag}: df_raw.shape = {df_raw.shape}")
    
    cols = list(df_raw.columns)
    if target not in cols:
        last_col = cols[-1]
        df_raw.rename(columns={last_col: target}, inplace=True)
        cols[-1] = target
    if 'date' not in cols:
        df_raw['date'] = pd.date_range(start='2024-01-01', periods=len(df_raw), freq='D')
        cols.append('date')
    cols.remove(target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [target]]
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    
    print(f"num_train = {num_train}, num_vali={num_vali}, num_test={num_test}")

    if features == 'M' or features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features == 'S':
        df_data = df_raw[[target]]

    print(f"df_data.shape = {df_data.shape}")
    
    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values
        
    
    print(f"data.shape = {data.shape}")

    data_x = data[border1:border2]
    data_y = data[border1:border2]

    print(save_data)
    if save_data:
        print("self.data_x.shape, self.data_y.shape, self.data_stamp.shape", data_x.shape, data_y.shape)
        import pickle
        p = f'/home/../multiTS/NFT/data/{dataset_name}/'
        print(p)
        if not os.path.exists(p): os.makedirs(p)
        with open(f"{p}{flag}_X.pkl", 'wb') as f: pickle.dump(data_x, f)
        with open(f"{p}{flag}_y.pkl", 'wb') as f: pickle.dump(data_y, f)
        print(f'Saved at:{p}')
