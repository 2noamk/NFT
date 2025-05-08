import sys

sys.path.append('/home/../multiTS/NFT/')
from dicts import data_to_steps, single_data_to_series_list, data_to_label_len, noaa_series_to_years, mini_electricity_list, melody_list
from data.proccess_data_functions import remove_outliers, data_to_raw_data_path, get_df, impute_data, standardize_data, get_datasets, save_to_pkl, plot_data


def process_data(data, lookback, horizon, series=None, year=None, with_date=True):
    path = f"/home/../multiTS/NFT/data/"
    
    df = remove_outliers(get_df(data_to_raw_data_path(data, series, year)))
    n = len(df)
    train_size, val_size, test_size = int(0.5*n), int(0.2*n), int(0.3*n)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:train_size + val_size + test_size]
            
    train_data, val_data, test_data = impute_data(train_df, val_df, test_df)
    
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_df=train_data, 
        val_df=val_data, 
        test_df=test_data, 
        lookback=lookback, 
        horizon=horizon,
        label_len=data_to_label_len.get(data, 0),
        with_date=True
        )
           
    X_train_standardized, y_train_standardized, X_val_standardized, y_val_standardized, X_test_standardized, y_test_standardized = standardize_data(
        train_dataset.X, train_dataset.y,
        val_dataset.X, val_dataset.y,
        test_dataset.X, test_dataset.y
    )
    
    
    print(f'X_train = {X_train_standardized.shape}, y_train = {y_train_standardized.shape}')
    print(f'X_val = {X_val_standardized.shape}, y_val = {y_val_standardized.shape}')
    print(f'X_test = {X_test_standardized.shape}, y_test = {y_test_standardized.shape}')
        
    if year is not None:
        pkl_path = f'{path}{data[0:4]}/years/{series}/{series}_{year}_{lookback}l_{horizon}h/'
        if with_date: date_stamp_path = f'{path}{data[0:4]}/years/{series}/{series}_date_stamp_{year}_{lookback}l_{horizon}h/'
    elif series is None:
        pkl_path = f'{path}{data}/{data}_{lookback}l_{horizon}h/'
        if with_date: date_stamp_path = f'{path}{data}/{data}_date_stamp_{lookback}l_{horizon}h/'
    else:
        pkl_path = f'{path}{data}/{series}/{series}_{lookback}l_{horizon}h/'
        if with_date: date_stamp_path = f'{path}{data}/{series}/{series}_date_stamp_{lookback}l_{horizon}h/'
    
    save_to_pkl(pkl_path,
        X_train_standardized, y_train_standardized, 
        X_val_standardized, y_val_standardized,
        X_test_standardized, y_test_standardized)
    
    if with_date: 
        save_to_pkl(date_stamp_path,
        train_dataset.X_mark, train_dataset.y_mark,
        val_dataset.X_mark, val_dataset.y_mark,
        test_dataset.X_mark, test_dataset.y_mark
        )
        
    print(f"saved to: {pkl_path}")

    # num_of_vars = data_to_num_vars_dict[data]
    
    # plot_data(X_train_standardized, lookback, num_of_vars)
    # plot_data(X_val_standardized, lookback, num_of_vars)
    # plot_data(X_test_standardized, lookback, num_of_vars)
    
    # plot_data(y_train_standardized, horizon, num_of_vars)
    # plot_data(y_val_standardized, horizon, num_of_vars)
    # plot_data(y_test_standardized, horizon, num_of_vars)




def main():
    data = 'noaa'
    if data == 'seasonal_trend_test':
         for seasonal_amplitude in [4, 5, 6, 7, 8]:#[0.05, 0.1, 0.15, 0.2]:
            for trend_amplitude in [0.2]:#[30, 40, 50, 60]:
                data = f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
                for step in data_to_steps['seasonal_trend_0_5']:
                    lookback, horizon = step
                    if lookback==150 and horizon ==50: break
                    process_data(
                        data=data,
                        lookback=lookback, 
                        horizon=horizon,
                        series=None,
                        with_date=True
                    )  
    if data in ['noaa', 'eeg_single', 'ecg_single', 'ett']:
        for series in single_data_to_series_list[data]:
            for step in data_to_steps[data]:
                lookback, horizon = step
                process_data(
                    data=data,
                    lookback=lookback, 
                    horizon=horizon,
                    series=series,
                    with_date=True
                )
    elif data == 'noaa_years':
        for lookback, horizon in data_to_steps[data]:
            for series in single_data_to_series_list[data]:
                for year in noaa_series_to_years[series]:
                    process_data(
                        data=f'noaa_{series}_{year}',
                        lookback=lookback, 
                        horizon=horizon,
                        series=series,
                        year=year
                    )       
    else:
        for step in data_to_steps[data]:
            lookback, horizon = step
            process_data(
                data=data,
                lookback=lookback, 
                horizon=horizon,
                series=None,
                with_date=True
            )  
    

if __name__ == "__main__":
    main()
    
