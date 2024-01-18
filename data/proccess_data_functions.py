from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import torch
import os

class dataset(Dataset):
    def __init__(self, data, lookback, horizon):
        self.sequence_length = lookback
        self.horizon = horizon
        self.X = []
        self.y = []
        
        for i in range(self.sequence_length, data.shape[0] - self.horizon):  
            self.X.append(data[i-self.sequence_length:i])
            self.y.append(data[i:i+self.horizon]) 

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_out


def get_df_without_outliers(path, n_cols=None):
    df = pd.read_pickle(path) 
    if n_cols: df = df.iloc[:, 0:3]
    return remove_outliers(df)


def impute_Data(train_df, val_df, test_df):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(train_df)
    
    train_data = torch.tensor(imputer.transform(train_df))
    val_data = torch.tensor(imputer.transform(val_df))
    test_data = torch.tensor(imputer.transform(test_df))

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
    
    
def get_datasets(train_df, val_df, test_df, lookback, predict_steps):
    train_dataset = dataset(train_df, lookback, predict_steps)
    val_dataset = dataset(val_df, lookback, predict_steps)
    test_dataset = dataset(test_df, lookback, predict_steps)
    return train_dataset, val_dataset, test_dataset


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

    _, axes = plt.subplots(num_cols, 1, figsize=(10, 4 * num_cols))  # Adjusting the size for better visibility

    for i, col in enumerate(df.columns):
        df[col].plot(ax=axes[i], title=col)
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()

    
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


def get_processed_data_many_series(dir_path, lookback, horizon, n_train, n_val, n_cols=None):
    filenames = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    filenames.sort()  # Sort the filenames for consistency

    train_filenames = filenames[:n_train]
    val_filenames = filenames[n_train:n_train + n_val]
    test_filenames = filenames[n_train + n_val:]

    def process_files(file_list):
        data = []
        for filename in file_list:
            file_path = os.path.join(dir_path, filename)
            df = get_df_without_outliers(file_path, n_cols=n_cols)
            df_tensor = torch.tensor(df.values, dtype=torch.float32)
            d = dataset(df_tensor, lookback, horizon)
            data.append((d.X, d.y))

        X = np.concatenate([x for x, _ in data], axis=0)
        y = np.concatenate([y for _, y in data], axis=0)
        return X, y

    x_train, y_train = process_files(train_filenames)
    x_val, y_val = process_files(val_filenames)
    x_test, y_test = process_files(test_filenames)

    return x_train, y_train, x_val, y_val, x_test, y_test

