import sys

sys.path.append('/home/noam.koren/multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_num_of_series, data_to_raw_data_path, data_to_steps, data_to_num_cols

from data.proccess_data_functions import get_processed_data_many_series, standardize_data, save_to_pkl


def process_data(data, lookback, horizon):
    
    path = f"/home/noam.koren/multiTS/NFT/data/{data}/"

    n_series = data_to_num_of_series[data]
    
    n_train = int(0.5 * n_series)
    n_val = int(0.15 * n_series)

    train_X, train_y, val_X, val_y, test_X, test_y = get_processed_data_many_series(
        dir_path=data_to_raw_data_path[data], 
        lookback=lookback, 
        horizon=horizon,
        n_train=n_train,
        n_val=n_val,
        n_cols=data_to_num_cols[data]
        )

    X_train_standardized, y_train_standardized, X_val_standardized, y_val_standardized, X_test_standardized, y_test_standardized = standardize_data(
    train_X, train_y,
    val_X, val_y,
    test_X, test_y
    )
    
    print(X_train_standardized.shape, y_train_standardized.shape, 
          X_val_standardized.shape, y_val_standardized.shape, 
          X_test_standardized.shape, y_test_standardized.shape)

    if data in ['ecg', 'eeg']:
        save_to_pkl(path + f"{data}_{lookback}l_{horizon}h_{n_series}series/",
        X_train_standardized, y_train_standardized, 
        X_val_standardized, y_val_standardized,
        X_test_standardized, y_test_standardized)
        
    else:
        save_to_pkl(path + f"{data}_{lookback}l_{horizon}h/",
            X_train_standardized, y_train_standardized, 
            X_val_standardized, y_val_standardized,
            X_test_standardized, y_test_standardized)
        
    num_of_vars = data_to_num_vars_dict[data]

    # plot_data(X_train_standardized, lookback, num_of_vars)
    # plot_data(X_val_standardized, lookback, num_of_vars)
    # plot_data(X_test_standardized, lookback, num_of_vars)
    
    # plot_data(y_train_standardized, horizon, num_of_vars)
    # plot_data(y_val_standardized, horizon, num_of_vars)
    # plot_data(y_test_standardized, horizon, num_of_vars)


def main():
    data = 'ecg'
    for step in data_to_steps[data]:
        lookback, horizon = step
        process_data(
            data=data,
            lookback=lookback, 
            horizon=horizon,
        )
    

if __name__ == "__main__":
    main()
    