import sys

sys.path.append('/home/noam.koren/multiTS/NFT/')
from lists import Stations, EEG, ETT, ECG, air_quality_seasonal_2_var_list, noaa_AEM00041217_years, noaa_AEM00041194_years, noaa_AE000041196_years, mini_electricity_list, melody_list, air_quality_seasonal_list

data_to_num_cols = {
    'ecg': None, 
    'ecg_single': None, 
    'eeg': None, 
    'eeg_single': None, 
    'chorales': None, 
    'noaa': None,
    'air_quality': None, 
    'cabs': None,
    'eeg_3_lead': 3,
    }


data_to_num_vars_dict = {
    'ecg': 12, 
    'ecg_single': 12, 
    'eeg': 36, 
    'eeg_3_lead': 3, 
    'eeg_single': 36, 
    'chorales': 6, 
    'melodies': 6, 
    'noaa': 3,
    'noaa_years': 3,
    'air_quality': 13, 
    'air_quality_seasonal': 4, 
    'air_quality_seasonal_2_var': 2, 
    'cabs': 4,
    'etth1' : 7,
    'etth2' : 7,
    'ettm1' : 7,
    'ettm2' : 7,
    'weather': 21,
    'traffic': 862,
    'electricity': 321,
    'mini_electricity': 2,
    'illness': 7,
    'exchange': 8,
    }


data_to_num_of_series = {
    'ecg': 600, 
    'ecg_single': 1, 
    'eeg': 32, 
    'eeg_3_lead': 32, 
    'eeg_single': 1, 
    'chorales': 100, 
    'melodies': 1, 
    'noaa': 1,
    'noaa_years': 1,
    'air_quality': 1,
    'air_quality_seasonal': 2,
    'air_quality_seasonal_2_var': 6,  
    'cabs': 536,
    'etth1': 1,
    'etth2': 1,
    'ettm1': 1,
    'ettm2': 1,
    'weather': 1,
    'traffic':1,
    'electricity':1,
    'mini_electricity': 1,
    'illness': 1,
    'exchange': 1,
    }


data_to_num_nft_blocks = {
    'ecg': 2, 
    'ecg_single': 3, 
    'eeg': 3, 
    'eeg_3_lead': 3, 
    'eeg_single': 3, 
    'chorales': 3, 
    'melodies': 2, 
    'noaa': 2,
    'noaa_years': 2,
    'air_quality': 3, 
    'air_quality_seasonal': 3,
    'air_quality_seasonal_2_var': 3, 
    'cabs': 2,
    'etth1': 2,
    'etth2': 2,
    'ettm1': 2,
    'ettm2': 2,
    'weather': 2,
    'traffic':2,
    'electricity':2,
    'mini_electricity': 2,
    'illness': 2,
    'exchange': 2,
    'seasonality': 2,
    'trend_noise': 2,
    'trend': 2,  
    }


data_to_steps = {
    'cabs': [
    (15, 1), (15, 3), (15, 5), 
    (30, 1), (30, 5), (30, 10), (30, 15), (30, 20),
        ],
    'air_quality' : [
    # (5, 1), 
    # (15, 1), (15, 3), (15, 5), 
    # (30, 1), (30, 5), (30, 10), (30, 15), (30, 20),
    (40, 1), (40, 5), (40, 10), (40, 15), (40, 20), (40, 25), (40, 30),
    # (50, 1), (50, 5), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30),
    # (60, 1), (60, 5), (60, 10), (60, 15), (60, 20), (60, 25), (60, 30),
        ],

    'air_quality_seasonal': [(40, 30)],
    'air_quality_seasonal_2_var': [(40, 30)], 
    'melodies' : [
        (5, 5)
    ],
    'chorales' : [
    # (5, 1), 
    (10, 1), (10, 2),
    (10, 3), (10, 4), (10, 5), 
    ],
    'noaa': [
    # (7, 1), 
    # (14, 1), (14, 7), 
    # (15, 1), (15, 7),
    # (30, 1), (30, 7), (30, 15),
    # (60, 1), (60, 7), (60, 15), (60, 30),
    # (90, 1), (90, 7), (90, 15), (90, 30), (90, 45),
    (360, 1), (360, 7), (360, 15), (360, 30), (360, 60), (360, 90),(360, 120), (360, 180),
    # (720, 1), (720, 7), (720, 15), (720, 30), (720, 60), (720, 180), (720, 360)
    ],
    'noaa_years': [
    # (5, 1),
    # (15, 1), 
    # (15, 5), 
    # (15, 10),
    (10, 15),
    # (15, 20),
    # (30, 1), 
    # (30, 5), 
    # (30, 10), 
    # (30, 15), 
    # (30, 20),
    # (60, 1), (60, 5), (60, 10), (60, 15), (60, 20),
    ],
    'ecg': [
    # (50, 1), (50, 10),
    # (100, 1), 
    # (100, 5), 
    # (100, 10), 
    # (100, 15), 
    # (100, 20), 
    # (100, 25), 
    # (100,30), 
    # (200, 1), (200, 5), (200, 10), (200, 15),  (200, 20), (200, 25),
    ],
    'eeg': [
    # (50, 1), (50, 10), 
    (100, 1), 
    # (100, 5),
    (100, 10), 
    # (100, 15), 
    (100, 20),
    # (200, 1), (200, 5), (200, 10), (200, 15), (200, 20),
    ],
    'eeg_3_lead': [
    (100, 1), 
    # (100, 5), 
    (100, 10), 
    # (100, 15), 
    (100, 20),
    ],
    'ecg_single': [
    # (50, 1), (50, 10),
    # (100, 1), (100, 10), (100, 25),
    (200, 1), (200, 10), (200, 25), (200, 50), 
    # (200, 100),
    # (300, 1), (300, 10), (300, 25), (300, 50), (300, 100)
    ],
    'eeg_single': [
    # (50, 1), (50, 10),
    (100, 1), 
    (100, 10), 
    (100, 25),
    # (200, 1), (200, 10), (200, 25),# (200, 50), (200, 100),
    # (300, 1), (300, 10), (300, 25), #(300, 50), (300, 100)
    ],
    'electricity': [
        # (96, 1),
        # (96, 16),
        (96, 32),
        # (96, 96), 
        # (96, 192), 
        # (96, 336), 
        # (96, 720)
                ],
    
    'mini_electricity': [
        (96, 720)
                ],
    'etth1': [
        (96, 1), 
        (96, 10), (96, 24), (96, 48),
        # (96, 96),
        # (96, 192), 
        # (96, 336),
        # (96, 720)
        ],
    'etth2': [
        (96, 1), 
        (96, 10), (96, 24), (96, 48), 
        #       (96, 96),
        # (96, 192), (96, 336), (96, 720)
        ],
    'ettm1': [
        (96, 1), 
        (96, 10), (96, 24), 
        (96, 48),
        # (96, 96),
        # (96, 192), 
        # (96, 336),
        # (96, 720)
        ],
    'ettm2': [
        (96, 1), 
        (96, 10), (96, 24), (96, 48), 
        (96, 96),
        # (96, 192), 
        # (96, 336),
        # (96, 720)
        ],
    'exchange': [
        (96, 1), 
        (96, 16),
        (96, 32), 
        (96, 48),
        # (96, 96), 
        # (96, 192), 
        # (96, 336), 
        # (96, 720)
        ],
    'illness': [ 
        # (36, 1),(36, 12), 
        (36, 24), (36, 36), (36, 48), (36, 60),
                ],
    'traffic': [
        # (96, 1), 
        (96, 16), 
        # (96, 32), 
        # (96, 48),
        # (96, 96), 
        # (96, 192), 
        # (96, 336), 
        # (96, 720)
        ],
    }


data_to_label_len = {
    'ecg_single': 0,
    'eeg_single': 0,
    'noaa': 0,
    'ecg': 0,
    'eeg': 0,
    'eeg_3_lead': 0,
    'air_quality': 0,   
    'air_quality_seasonal': 0,
    'air_quality_seasonal_2_var': 0,
    'chorales' : 0,
    'melodies' : 0,
    # 'electricity': 48,
    'mini_electricity': 0,
    'electricity': 0,
    'etth1': 0,
    'etth2': 0,
    'ettm1': 0,
    'ettm2': 0,
    # 'etth1': 48,
    # 'etth2': 48,
    # 'ettm1': 48,
    # 'ettm2': 48,
    'exchange': 0,
    # 'exchange': 48,
    # 'illness': 0,
    'illness': 18,
    'traffic': 0,
    # 'traffic': 48,
    'weather': 48,
    }


single_data_to_series_list = {
    'ecg_single': ECG,
    'eeg_single': EEG,
    'noaa': Stations,
    'noaa_years': Stations,
    'ecg': [None],
    'eeg': [None],
    'air_quality': [None],
    'ett': ETT,
    'weather': [None],
    'mini_electricity': mini_electricity_list,
    'air_quality_seasonal': air_quality_seasonal_list,
    'air_quality_seasonal_2_var': air_quality_seasonal_2_var_list,
    }


noaa_series_to_years = {
    'AEM00041194': noaa_AEM00041194_years,
    'AE000041196': noaa_AE000041196_years,
    'AEM00041217': noaa_AEM00041217_years,
    }
