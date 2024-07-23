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
    def __init__(self, data, lookback, horizon, label_len=0, data_stamp=None):
        self.sequence_length = lookback
        self.horizon = horizon
        self.label_len = label_len
        self.X = []
        self.y = []
        self.add_date = True if data_stamp is not None else False
        if self.add_date:
            self.X_mark = []
            self.y_mark = []

        for i in range(self.sequence_length, data.shape[0] - self.horizon):  
            self.X.append(data[i - self.sequence_length:i])
            self.y.append(data[i - self.label_len:i + self.horizon]) 
            if self.add_date:
                self.X_mark.append(data_stamp[i - self.sequence_length:i])
                self.y_mark.append(data_stamp[i - self.label_len:i + self.horizon]) 

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
   
   
def data_to_raw_data_path(data, series=None, year=None):
    base_path = "/home/noam.koren/multiTS/NFT/data/"
    path_mappings = {
        'noaa': lambda: f'{base_path}noaa/noaa_ghcn/noaa_pkl/{series}.pkl',
        'ecg_single': lambda: f'{base_path}ecg/pkl_files/{series}.pkl',
        'eeg_single': lambda: f'{base_path}eeg/eval_normal_pkl/{series}.pkl',
        'noaa_year': lambda: f'{base_path}noaa/noaa_ghcn/years/embedded/{series}/{series}_{year}.pkl',
        'ecg': lambda: f'{base_path}ecg/pkl_files', 
        'eeg': lambda: f'{base_path}eeg/eval_normal_pkl',
        'eeg_3_lead': lambda: f'{base_path}eeg/eval_normal_pkl',
        'chorales': lambda: f'{base_path}chorales/chorales_pkl', 
        'cabs': lambda: f'{base_path}cabs/cabs_150_pkl',
        'electricity': lambda: f'{base_path}autoformer_datasets/{data}/{data}.csv',
        'etth1': lambda: f'{base_path}autoformer_datasets/ETT-small/ETTh1.csv',
        'etth2': lambda: f'{base_path}autoformer_datasets/ETT-small/ETTh2.csv',
        'ettm1': lambda: f'{base_path}autoformer_datasets/ETT-small/ETTm1.csv',
        'ettm2': lambda: f'{base_path}autoformer_datasets/ETT-small/ETTm2.csv',
        'exchange_rate': lambda: f'{base_path}autoformer_datasets/{data}/{data}.csv',
        'illness': lambda: f'{base_path}autoformer_datasets/{data}/national_illness.csv',
        'traffic': lambda: f'{base_path}autoformer_datasets/{data}/{data}.csv',
        'weather':lambda: f'{base_path}autoformer_datasets/{data}/{data}.csv',
    }

    # Check if the 'data' key starts with 'noaa' and the year is specified
    if data.startswith('noaa') and year is not None:
        data = 'noaa_year'
    
    # Default path if data key not found in the dictionary
    default_path = f'{base_path}{data}/{data}.pkl'
    
    # Fetch the appropriate path using the dictionary, or use the default if no matching key
    df_path = path_mappings.get(data, lambda: default_path)()
    print(df_path)
    return df_path

    
def remove_outliers(df):
    Q1 = df.quantile(0.25, numeric_only=True)
    Q3 = df.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1

    df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_out


def get_df_without_outliers(path, n_cols=None):
    # Determine the file extension and read the file accordingly
    if path.endswith('.pkl'):  df = pd.read_pickle(path)
    elif path.endswith('.csv'):  df = pd.read_csv(path)
    else: raise ValueError("Unsupported file type. Please use a .csv or .pkl file.")
    if n_cols: df = df.iloc[:, 0:n_cols]
    return remove_outliers(df)


def impute_data(train_df, val_df, test_df):
    # Identify numeric columns (excluding date or categorical data)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    # Create an imputer instance for numeric data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Fit the imputer only on the numeric columns of the training data
    imputer.fit(train_df[numeric_cols])

    # Function to impute and reconstruct DataFrame
    def impute_and_reconstruct(df):
        # Impute numeric columns
        imputed_data = imputer.transform(df[numeric_cols])
        # Construct the new DataFrame with imputed numeric data and original non-numeric data
        new_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
        # Reattach non-numeric columns to the DataFrame
        non_numeric = df.drop(columns=numeric_cols)
        return pd.concat([new_df, non_numeric], axis=1)

    # Impute and reconstruct each dataset
    train_data = impute_and_reconstruct(train_df)
    val_data = impute_and_reconstruct(val_df)
    test_data = impute_and_reconstruct(test_df)

    return train_data, val_data, test_data



def standardize_data(X_train, y_train, X_val, y_val, X_test, y_test):
    _, _, num_features = X_train.shape

    X_scaler = StandardScaler().fit(X_train.reshape(-1, num_features))
    
    X_train_standardized = X_scaler.transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_standardized = X_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_standardized = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, num_features))
    
    y_train_standardized = y_scaler.transform(y_train.reshape(-1, num_features)).reshape(y_train.shape)
    y_val_standardized = y_scaler.transform(y_val.reshape(-1, num_features)).reshape(y_val.shape)
    y_test_standardized = y_scaler.transform(y_test.reshape(-1, num_features)).reshape(y_test.shape)
    
    return X_train_standardized, y_train_standardized, X_val_standardized, y_val_standardized, X_test_standardized, y_test_standardized


def save_to_pkl(p, train_dataset_X, train_dataset_y, val_dataset_X, val_dataset_y, test_dataset_X, test_dataset_y):
    if not os.path.exists(p):
        os.makedirs(p)
    with open(p + "train_X.pkl", 'wb') as f:
        pickle.dump(train_dataset_X, f)
    with open(p + "train_y.pkl", 'wb') as f:
        pickle.dump(train_dataset_y, f)
    with open(p + "val_X.pkl", 'wb') as f:
        pickle.dump(val_dataset_X, f)
    with open(p + "val_y.pkl", 'wb') as f:
        pickle.dump(val_dataset_y, f)
    with open(p + "test_X.pkl", 'wb') as f:
        pickle.dump(test_dataset_X, f)
    with open(p + "test_y.pkl", 'wb') as f:
        pickle.dump(test_dataset_y, f)


def plot_data(data, data_len, n_series):
    """
    function takes the first time stamp predicted for every sample and plots it.
    i.e; function does not plot later horizon points.
    """

    actual_series = data.reshape(-1, data_len, n_series)[:, 0, :]

    n_cols = 4
    n_rows = int(n_series / n_cols) + 1
    plt.figure(figsize=(50, 25))

    for series in range(n_series):
        plt.subplot(n_rows, n_cols, series + 1)
        plt.plot(actual_series[:, series], label="Actual")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Series {series + 1}")

    plt.suptitle("Actual vs Predicted Time Series")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


def plot_df(df):
    num_cols = df.shape[1]
    num_rows = (num_cols + 4) // 5  # Calculate the number of rows needed, rounding up

    fig, axes = plt.subplots(num_rows, min(num_cols, 5), figsize=(20, 4 * num_rows))  # Adjusting the size for better visibility
        
    for i, col in enumerate(df.columns):
        row, col_num = divmod(i, 5)
        if num_rows > 1:  # If there are multiple rows, access the subplot as a 2D array
            ax = axes[row][col_num]
        else:  # If there is only one row, access the subplot as a 1D array
            ax = axes[col_num]
        df[col].plot(ax=ax, title=col)
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()

 
def process_date(df):
    df_stamp = df[['date']]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)

    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(['date'], axis=1).values

    return data_stamp

  
def get_poccesed_data(dir_path, lookback, horizon, n_rows=None):
    data = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(dir_path, filename)
            df = get_df_without_outliers(file_path, n_rows)
            df_tensor = torch.tensor(df.values, dtype=torch.float32)
            d = dataset(df_tensor, lookback, horizon)
            data.append((d.X, d.y))

    X = np.concatenate([x for x, _ in data], axis=0)
    y = np.concatenate([y for _, y in data], axis=0)
    return X, y
   
    
def get_dataset_with_date(df, lookback, horizon, label_len):
    cols = list(df.columns)
    if 'date' in cols: cols.remove('date')
    else: df['date'] = datetime.now().strftime('%Y-%m-%d')
    
    df_data = df[['date'] + cols]# + [target]]
    
    df_stamp = df_data[['date']]
    data_stamp = process_date(df_stamp)
        
    cols_data = df_data.columns[1:]
    df_data = df_data[cols_data]
    
    df_data = df_data.apply(pd.to_numeric, errors='coerce')

    ts = torch.tensor(df_data.values, dtype=torch.float32)
    d = dataset(data=ts,
                lookback=lookback,
                horizon=horizon, 
                label_len=label_len, 
                data_stamp=data_stamp
                )
    return d
 
       
def get_datasets(train_df, val_df, test_df, lookback, horizon, label_len, with_date=True):
    if not with_date:
        if 'date' in list(train_df.columns): 
            train_df = train_df.drop('date', axis=1)
            val_df = val_df.drop('date', axis=1)
            test_df = test_df.drop('date', axis=1)
        train_dataset = dataset(data=train_df,
                                lookback=lookback,
                                horizon=horizon, 
                                label_len=label_len
                                )
        val_dataset = dataset(data=val_df, 
                              lookback=lookback, 
                              horizon=horizon, 
                              label_len=label_len
                              )
        test_dataset = dataset(data=test_df, 
                               lookback=lookback, 
                               horizon=horizon, 
                               label_len=label_len
                               )
    else:
        train_dataset = get_dataset_with_date(train_df, lookback, horizon, label_len)
        val_dataset = get_dataset_with_date(val_df, lookback, horizon, label_len)
        test_dataset = get_dataset_with_date(test_df, lookback, horizon, label_len)
    return train_dataset, val_dataset, test_dataset


def get_processed_data_many_series(dir_path, lookback, horizon, label_len, n_train, n_val, n_cols=None, with_date=False):
    filenames = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    filenames.sort()  # Sort the filenames for consistency

    train_filenames = filenames[:n_train]
    val_filenames = filenames[n_train:n_train + n_val]
    test_filenames = filenames[n_train + n_val:]

    def process_files(file_list):
        data = []
        for filename in file_list:
            print(f"file={filename}")
            file_path = os.path.join(dir_path, filename)
            df = get_df_without_outliers(file_path, n_cols=n_cols)
            print(df.head())
            df_tensor = torch.tensor(df.values, dtype=torch.float32)
            d = dataset(df_tensor, lookback, horizon)
            data.append((d.X, d.y))

        X = np.concatenate([x for x, _ in data], axis=0)
        y = np.concatenate([y for _, y in data], axis=0)
        return X, y
    
    def process_files_with_date_stamp(file_list):
        data = []
        for filename in file_list:
            print(f"file={filename}")
            file_path = os.path.join(dir_path, filename)
            df = get_df_without_outliers(file_path, n_cols=n_cols)
            
            d = get_dataset_with_date(df, lookback, horizon, label_len)
            
            data.append((d.X, d.y, d.X_mark, d.y_mark))

        X = np.concatenate([x for x, _, _, _ in data], axis=0)
        y = np.concatenate([y for _, y, _, _ in data], axis=0)
        data_stamp_X = np.concatenate([x_mark for _, _, x_mark, _ in data], axis=0)
        data_stamp_y = np.concatenate([y_mark for _, _, _, y_mark in data], axis=0)
        
        return X, y, data_stamp_X, data_stamp_y
    
    if not with_date:
        x_train, y_train = process_files(train_filenames)
        x_val, y_val = process_files(val_filenames)
        x_test, y_test = process_files(test_filenames)
        
        return x_train, y_train, x_val, y_val, x_test, y_test, None, None, None
    else:
        X_train, y_train, train_stamp_X, train_stamp_y = process_files_with_date_stamp(train_filenames)
        X_val, y_val, val_stamp_X, val_stamp_y = process_files_with_date_stamp(val_filenames)
        X_test, y_test, test_stamp_X, test_stamp_y = process_files_with_date_stamp(test_filenames)        
        
        return X_train, y_train, X_val, y_val, X_test, y_test, train_stamp_X, train_stamp_y, val_stamp_X, val_stamp_y, test_stamp_X, test_stamp_y
