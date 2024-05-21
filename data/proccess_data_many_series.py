import sys

sys.path.append('/home/noam.koren/multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_num_of_series, data_to_steps, data_to_num_cols, data_to_label_len

from data.proccess_data_functions import data_to_raw_data_path, get_processed_data_many_series, standardize_data, save_to_pkl


def process_data(data, lookback, horizon, with_date=True):
    
    path = f"/home/noam.koren/multiTS/NFT/data/{data}/"

    n_series = data_to_num_of_series[data]
    n_train = int(0.5 * n_series)
    n_val = int(0.2 * n_series)
    
    pocessed_data = get_processed_data_many_series(
        dir_path=data_to_raw_data_path(data), 
        lookback=lookback, 
        horizon=horizon,
        label_len=data_to_label_len[data],
        n_train=n_train,
        n_val=n_val,
        n_cols=data_to_num_cols[data],
        with_date=with_date,
        )
    
    train_X, train_y, val_X, val_y, test_X, test_y, _, _, _, _, _, _ = pocessed_data
    
    if with_date:
        _, _, _, _, _, _, train_stamp_X, train_stamp_y, val_stamp_X, val_stamp_y, test_stamp_X, test_stamp_y = pocessed_data

    X_train_standardized, y_train_standardized, X_val_standardized, y_val_standardized, X_test_standardized, y_test_standardized = standardize_data(
    train_X, train_y,
    val_X, val_y,
    test_X, test_y
    )
    
    print(
        X_train_standardized.shape, y_train_standardized.shape, 
        X_val_standardized.shape, y_val_standardized.shape, 
        X_test_standardized.shape, y_test_standardized.shape,
        )

    if data in ['ecg', 'eeg']:
        save_to_pkl(
            path + f"{data}_{lookback}l_{horizon}h_{n_series}series/",
            X_train_standardized, y_train_standardized, 
            X_val_standardized, y_val_standardized,
            X_test_standardized, y_test_standardized
        )
        if with_date:
            save_to_pkl(
                path + f"{data}_date_stamp_{lookback}l_{horizon}h_{n_series}series/",
                train_stamp_X, train_stamp_y, 
                val_stamp_X, val_stamp_y, 
                test_stamp_X, test_stamp_y
            )
        
    else:
        save_to_pkl(
            path + f"{data}_{lookback}l_{horizon}h/",
            X_train_standardized, y_train_standardized, 
            X_val_standardized, y_val_standardized,
            X_test_standardized, y_test_standardized
            )
        if with_date:
            save_to_pkl(
                path + f"{data}_date_stamp_{lookback}l_{horizon}h/",
                train_stamp_X, train_stamp_y, 
                val_stamp_X, val_stamp_y, 
                test_stamp_X, test_stamp_y
                )
        
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
            with_date=True
        )
    

if __name__ == "__main__":
    main()
    