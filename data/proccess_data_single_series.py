import sys

sys.path.append('/home/noam.koren/multiTS/NFT/')
from dicts import data_to_num_vars_dict, data_to_steps, single_data_to_series_list
from lists import noaa_years
from data.proccess_data_functions import get_df_without_outliers, impute_Data, standardize_data, get_datasets, save_to_pkl, plot_data


def process_data(data, lookback, horizon, series=None, year=None):
    path = f"/home/noam.koren/multiTS/NFT/data/"
    
    if data == 'noaa':
        df = get_df_without_outliers(
            f'{path}noaa/noaa_ghcn/noaa_pkl/{series}.pkl')
    elif data == 'ett':
        df = get_df_without_outliers(
            f'{path}ett/pkl_files/{series}.pkl')
    elif data == 'weather':
        df = get_df_without_outliers(
            f'{path}weather/weather_no_date.pkl')
    elif data == 'ecg_single':
        df = get_df_without_outliers(
            f'{path}ecg/csv_files/{series}.pkl')
    elif data == 'eeg_single':
        df = get_df_without_outliers(
            f'{path}eeg/eval_normal_pkl/{series}.pkl')
    elif data[0:4] == 'noaa':
        print(series, year)
        df = get_df_without_outliers(
            f'{path}noaa/noaa_ghcn/years/embedded/{series}/{series}_{year}.pkl')
    else:
        df = get_df_without_outliers(path + f'{data}/{data}.pkl') 
           
    n = len(df)
    train_size, val_size, test_size = int(0.5*n), int(0.2*n), int(0.3*n)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:train_size + val_size + test_size]
            
    train_data, val_data, test_data = impute_Data(train_df, val_df, test_df)
    
    train_dataset, val_dataset, test_dataset = get_datasets(train_data, val_data, test_data, lookback, horizon)
           
    X_train_standardized, y_train_standardized, X_val_standardized, y_val_standardized, X_test_standardized, y_test_standardized = standardize_data(
        train_dataset.X, train_dataset.y,
        val_dataset.X, val_dataset.y,
        test_dataset.X, test_dataset.y
    )
        
    
    if year is not None:
        pkl_path = f'{path}{data[0:4]}/years/{series}/{series}_{year}_{lookback}l_{horizon}h/'
    elif series is None:
        pkl_path = f'{path}{data}/{data}_{lookback}l_{horizon}h/'
    else:
        pkl_path = f'{path}{data}/{series}/{series}_{lookback}l_{horizon}h/'
    
    save_to_pkl(pkl_path,
        X_train_standardized, y_train_standardized, 
        X_val_standardized, y_val_standardized,
        X_test_standardized, y_test_standardized)
    
    print(f"saved to: {pkl_path}")

    # num_of_vars = data_to_num_vars_dict[data]
    
    # plot_data(X_train_standardized, lookback, num_of_vars)
    # plot_data(X_val_standardized, lookback, num_of_vars)
    # plot_data(X_test_standardized, lookback, num_of_vars)
    
    # plot_data(y_train_standardized, horizon, num_of_vars)
    # plot_data(y_val_standardized, horizon, num_of_vars)
    # plot_data(y_test_standardized, horizon, num_of_vars)


def main():
    data = 'weather'
    if data in ['noaa', 'eeg_single', 'ecg_single', 'ett']:
        for series in single_data_to_series_list[data]:
            for step in data_to_steps[data]:
                lookback, horizon = step
                process_data(
                    data=data,
                    lookback=lookback, 
                    horizon=horizon,
                    series=series
                )
    elif data == 'noaa_years':
        for step in data_to_steps[data]:
            for series in  single_data_to_series_list[data[0:4]]:
                lookback, horizon = step
                for year in noaa_years:
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
                series=None
            )  
    

if __name__ == "__main__":
    main()
    
