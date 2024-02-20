import sys

sys.path.append('NFT/')
from lists import Stations, Ecg, EEG

data_to_num_cols = {
    'ecg': None, 
    'ecg_single': None, 
    'eeg': 3, 
    'eeg_single': None, 
    'chorales': None, 
    'noaa': None,
    'air_quality': None, 
    'cabs': None
    }


data_to_num_vars_dict = {
    'ecg': 12, 
    'ecg_single': 12, 
    'eeg': 3, 
    'eeg_single': 36, 
    'chorales': 6, 
    'noaa': 3,
    'air_quality': 13, 
    'cabs': 4
    }


data_to_num_of_series = {
    'ecg': 600, 
    'ecg_single': 1, 
    'eeg': 32, 
    'eeg_single': 1, 
    'chorales': 100, 
    'noaa': 1,
    'air_quality': 1, 
    'cabs': 536
    }


data_to_raw_data_path = {
    'ecg': 'NFT/data/ecg/600_pkl_files', 
    'eeg': '/home/noam.koren/multiTS/NFT/data/eeg/eval_normal_pkl',
    'chorales': 'NFT/data/chorales/chorales_pkl', 
    'cabs': '/home/noam.koren/multiTS/NFT/data/cabs/cabs_150_pkl'
    }


data_to_steps = {
    'cabs': [
    (15, 1), (15, 3), (15, 5), 
    (30, 1), (30, 5), (30, 10), (30, 15), (30, 20),
    ],
    'air_quality' : [
    (5, 1), 
    (15, 1), (15, 3), (15, 5), 
    (30, 1), (30, 5), (30, 10), (30, 15), (30, 20),
    (40, 1), (40, 5), (40, 10), (40, 15), (40, 20), (40, 25), (40, 30),
    (50, 1), (50, 5), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30),
    (60, 1), (60, 5), (60, 10), (60, 15), (60, 20), (60, 25), (60, 30),
    ],
    'chorales' : [
    (5, 1), 
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), 
    ],
    'noaa': [
    (7, 1), 
    (14, 1), (14, 7), 
    (15, 1), (15, 7),
    (30, 1), (30, 7), (30, 15),
    (60, 1), (60, 7), (60, 15), (60, 30),
    (90, 1), (90, 7), (90, 15), (90, 30), (90, 45),
    (360, 1), (360, 7), (360, 15), (360, 30), (360, 60), (360, 90),(360, 120), (360, 180),
    (720, 1), (720, 7), (720, 15), (720, 30), (720, 60), (720, 180), (720, 360)
    ],
    'ecg': [
    (50, 1), (50, 10),
    (100, 1), (100, 5), (100, 10), (100, 15), (100, 20), 
    (100, 25), (100,30), 
    (200, 1), (200, 10), (200, 25),
    ],
    'eeg': [
    (100, 1), (100, 10), (100, 20),
    ],
    'ecg_single': [
    (50, 1), (50, 10),
    (100, 1), (100, 10), (100, 25),
    (200, 1), (200, 10), (200, 25), (200, 50), (200, 100),
    (300, 1), (300, 10), (300, 25), (300, 50), (300, 100)
    ],
    'eeg_single': [
    (50, 1), (50, 10),
    (100, 1), (100, 10), (100, 25),
    (200, 1), (200, 10), (200, 25), (200, 50), (200, 100),
    (300, 1), (300, 10), (300, 25), (300, 50), (300, 100)
    ],
}


single_data_to_series_list = {
    'ecg_single': Ecg,
    'eeg_single': EEG,
    'noaa': Stations,
    'ecg': [None],
    'eeg': [None],
    'air_quality': [None],
    
}