import sys

sys.path.append('/home/../multiTS/NFT/')

from lists import Stations
import pandas as pd
import pickle
import os

base_dir = '/home/../multiTS/NFT/data/noaa/noaa_ghcn/'

for station in Stations:
    output_dir = f'/home/../multiTS/NFT/data/noaa/noaa_ghcn/years/original/{station}/'

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    with open(f'{base_dir}original_noaa_files/{station}.csv', 'rb') as file:
        df =  pd.read_csv(file)
    print(df.head())
    print(df.columns)
    # Ensure the date column is in datetime format
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)

    # Step 2 and 3: Group the data by year
    grouped = df.groupby(df['DATE'].dt.year)

    # Step 4: Write separate Pickle files for each year
    for year, data in grouped:
        filename = f'{output_dir}{station}_{year}.pkl'  # Naming each file as 'data_YEAR.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(data, file) 
        d = pd.read_pickle(filename)
        print(f'{year},')
