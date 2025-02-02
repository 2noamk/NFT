import pandas as pd
from collections import defaultdict
import sys
sys.path.append('/home/noam.koren/multiTS/NFT/')
from dicts import single_data_to_series_list

# Function to calculate min values for each horizon
def calculate_min_per_horizon(data):
    grouped = data.groupby('Horizon')
    horizon_min_values = {}
    for horizon, group in grouped:
        min_test_mse_row = group.loc[group['test_mse'].idxmin()]  # Row with minimum test_mse
        min_test_mae_row = group.loc[group['test_mae'].idxmin()]  # Row with minimum test_mae

        horizon_min_values[horizon] = {
            "min_test_mse": {
                "value": min_test_mse_row['test_mse'],
                "corresponding_test_mae": min_test_mse_row['test_mae']
            },
            "min_test_mae": {
                "value": min_test_mae_row['test_mae'],
                "corresponding_test_mse": min_test_mae_row['test_mse']
            }
        }
    return horizon_min_values

# Function to calculate min values across series for each horizon
def calculate_min_per_horizon_across_series(dataset, series_list):
    horizon_min_aggregated = defaultdict(list)
    for series in series_list:
        file_path = f"/home/noam.koren/multiTS/NFT/results/{dataset}/nft/nft_{dataset}_{series}_None_results.xlsx"
        data = pd.read_excel(file_path)
        series_horizon_min = calculate_min_per_horizon(data)
        for horizon, min_values in series_horizon_min.items():
            horizon_min_aggregated[horizon].append(min_values)

    # Find global min per horizon
    global_min_per_horizon = {}
    for horizon, min_values_list in horizon_min_aggregated.items():
        min_test_mse = min(min_values_list, key=lambda x: x["min_test_mse"]["value"])
        min_test_mae = min(min_values_list, key=lambda x: x["min_test_mae"]["value"])
        global_min_per_horizon[horizon] = {
            "min_test_mse": min_test_mse["min_test_mse"],
            "min_test_mae": min_test_mae["min_test_mae"]
        }

    return global_min_per_horizon

def get_min_vals():
    # For datasets without series splitting
    for dataset in ['air_quality', 'chorales', 'ettm1', 'traffic', 'electricity', 'exchange']:
        file_path = f"/home/noam.koren/multiTS/NFT/results/{dataset}/nft/nft_{dataset}_results.xlsx"
        data = pd.read_excel(file_path)
        min_values_per_horizon = calculate_min_per_horizon(data)
        print(f"Min values per horizon for {dataset}:")
        for horizon, values in min_values_per_horizon.items():
            print(f"  Horizon {horizon}:")
            print(f"    Minimum test_mse: {values['min_test_mse']['value']} (Corresponding test_mae: {values['min_test_mse']['corresponding_test_mae']})")
            print(f"    Minimum test_mae: {values['min_test_mae']['value']} (Corresponding test_mse: {values['min_test_mae']['corresponding_test_mse']})")

    # For datasets with multiple series
    for dataset in ['noaa', 'ecg_single', 'eeg_single']:
        series_list = single_data_to_series_list[dataset]
        min_values_per_horizon = calculate_min_per_horizon_across_series(dataset, series_list)
        print(f"Aggregated Min values per horizon for {dataset}:")
        for horizon, values in min_values_per_horizon.items():
            print(f"  Horizon {horizon}:")
            print(f"    Minimum test_mse: {values['min_test_mse']['value']} (Corresponding test_mae: {values['min_test_mse']['corresponding_test_mae']})")
            print(f"    Minimum test_mae: {values['min_test_mae']['value']} (Corresponding test_mse: {values['min_test_mae']['corresponding_test_mse']})")

mse_data = {
        "ECL": {
            1: [0.051, 0.054],
            16: [0.080, 0.100],
            32: [0.141, 0.148],
        },
        "Exchange": {
            96: [0.060, 0.081],
            192: [0.164, 0.176],
            336: [0.422, 0.313],
            720: [0.670, 0.668],
        },
        "Traffic": {
            1: [0.200, 0.217],
            16: [0.320, 0.381],
            32: [0.390, 0.418],
            48: [0.400, 0.430],
        },
        "ETTm1": {
            48: [0.270, 0.280],
            96: [0.241, 0.287],
            192: [0.320, 0.327],
            336: [0.410, 0.355],
        },
        "Weather": {
            15: [0.301, 0.330],
            30: [0.313, 0.350],
            60: [0.325, 0.370],
            90: [0.330, 0.370],
        },
        "ECG": {
            1: [0.080, 0.085],
            10: [0.172, 0.247],
            25: [0.258, 0.380],
            100: [0.393, 0.460],
        },
        "EEG": {
            1: [0.075, 0.108],
            10: [0.159, 0.320],
            25: [0.238, 0.406],
        },
        "Chorales": {
            1: [0.223, 0.270],
            2: [0.242, 0.25], 
            3: [0.259, 0.250],
            4: [0.260, 0.265],
        },
        "Air_Quality": {
            5: [0.513, 0.580],
            10: [0.641, 0.655],
            15: [0.717, 0.72],
            25: [0.779, 0.78],
            },
        }


mse_fourier_data = {
        "ECL": {
            1: [0.051, 0.054],
            16: [0.080, 0.100],
            32: [0.141, 0.148],
        },
        "Exchange": {
            96: [0.060, 0.081],
            192: [0.164, 0.177],
            336: [0.422, 0.327],
            720: [0.670, 0.668],
        },
        "Traffic": {
            1: [0.200, 0.217],
            16: [0.320, 0.381],
            32: [0.390, 0.418],
            48: [0.400, 0.430],
        },
        "ETTm1": {
            48: [0.270, 0.293],
            96: [0.241, 0.287],
            192: [0.320, 0.327],
            336: [0.410, 0.355],
        },
        "Weather": {
            15: [0.301, 0.343],
            30: [0.313, 0.360],
            60: [0.325, 0.370],
            90: [0.330, 0.370],
        },
        "ECG": {
            1: [0.080, 0.085],
            10: [0.172, 0.247],
            25: [0.258, 0.380],
            100: [0.393, 0.460],
        },
        "EEG": {
            1: [0.075, 0.108],
            10: [0.159, 0.320],
            25: [0.238, 0.406],
        },
        "Chorales": {
            1: [0.223, 0.290],
            2: [0.242, 0.25], 
            3: [0.259, 0.250],
            4: [0.260, 0.265],
        },
        "Air_Quality": {
            5: [0.513, 0.586],
            10: [0.641, 0.655],
            15: [0.717, 0.731],
            25: [0.779, 0.797],
            },
        }


mae_data = {
        "ECL": {
            1: [0.146, 0.150],
            16: [0.219, 0.221],
            32: [0.250, 0.240],
        },
        "Exchange": {
            96: [0.173, 0.200],
            192: [0.290, 0.293],
            336: [0.478, 0.397],
            720: [0.684, 0.601],
        },
        "Traffic": {
            1: [0.190, 0.200],
            16: [0.270, 0.270],
            32: [0.330, 0.290],
            48: [0.410, 0.300],
        },
        "ETTm1": {
            48: [0.310, 0.340],
            96: [0.307, 0.310],
            192: [0.378, 0.330],
            336: [0.439, 0.370],
        },
        "Weather": {
            15: [0.420, 0.420],
            30: [0.447, 0.433],
            60: [0.430, 0.440],
            90: [0.430, 0.443],
        },
        "ECG": {
            1: [0.204, 0.240],
            10: [0.305, 0.370], 
            25: [0.375, 0.443],
            100: [0.481, 0.613],
        },
        "EEG": {
            1: [0.101, 0.155],
            10: [0.194, 0.345],
            25: [0.260, 0.405],
        },
        "Chorales": {
            1: [0.199, 0.209],
            2: [0.204, 0.218], 
            3: [0.219, 0.220],
            4: [0.226, 0.227],
        },
        "Air_Quality": {
            5: [0.495, 0.520],
            10: [0.530, 0.540],
            15: [0.550, 0.570],
            25: [0.600, 0.610],
            },
        }

def get_precentage_imp(data):
    # Input data
    mse_data = {
        "ECL": {
            1: [0.051, 0.054],
            16: [0.080, 0.100],
            32: [0.141, 0.148],
        },
        "Exchange": {
            96: [0.060, 0.081],
            192: [0.164, 0.176],
            336: [0.422, 0.313],
            720: [0.670, 0.668],
        },
        "Traffic": {
            1: [0.200, 0.217],
            16: [0.320, 0.381],
            32: [0.390, 0.418],
            48: [0.400, 0.430],
        },
        "ETTm1": {
            48: [0.270, 0.280],
            96: [0.241, 0.287],
            192: [0.320, 0.327],
            336: [0.410, 0.355],
        },
        "Weather": {
            15: [0.301, 0.330],
            30: [0.313, 0.350],
            60: [0.325, 0.370],
            90: [0.330, 0.370],
        },
        "ECG": {
            1: [0.080, 0.085],
            10: [0.172, 0.247],
            25: [0.258, 0.380],
            100: [0.393, 0.460],
        },
        "EEG": {
            1: [0.075, 0.108],
            10: [0.159, 0.320],
            25: [0.238, 0.406],
        },
        "Chorales": {
            1: [0.223, 0.270],
            2: [0.242, 0.25], 
            3: [0.259, 0.250],
            4: [0.260, 0.265],
        },
        "Air_Quality": {
            5: [0.513, 0.580],
            10: [0.641, 0.655],
            15: [0.717, 0.72],
            25: [0.779, 0.78],
            },
        }

    # Conduct t-tests and calculate improvement percentages
    results = {}

    for dataset, entries in data.items():
        improvements = []
        values1 = []
        values2 = []

        for key, values in entries.items():
            val1, val2 = values
            improvement = ((val2 - val1) / val1) * 100
            improvements.append(improvement)
            values1.append(val1)
            values2.append(val2)

        # Average improvement
        average_improvement = sum(improvements) / len(improvements)
        results[dataset] = average_improvement


    # Print the results
    for dataset, avg_improvement in results.items():
        print(f"{dataset}: Average Improvement = {avg_improvement:.2f}%")

def get_precentage_imp_mse():
    print('MSE')
    get_precentage_imp(mse_data)


def get_precentage_imp_mse_fourier():
    print('mse_fourier')
    get_precentage_imp(mse_fourier_data)

def get_precentage_imp_mae():
    print('MAE')
    get_precentage_imp(mae_data)

get_precentage_imp_mse_fourier()

"""
MSE
ECL: Average Improvement = 11.95%
Exchange: Average Improvement = 4.05%
Traffic: Average Improvement = 10.56%
ETTm1: Average Improvement = 2.89%
Weather: Average Improvement = 11.86%
ECG: Average Improvement = 28.55%
EEG: Average Improvement = 71.95%
Chorales: Average Improvement = 5.71%
Air_Quality: Average Improvement = 3.95%
"""

"""
MAE
ECL: Average Improvement = -0.12%
Exchange: Average Improvement = -3.11%
Traffic: Average Improvement = -8.42%
ETTm1: Average Improvement = -4.44%
Weather: Average Improvement = 0.55%
ECG: Average Improvement = 21.13%
EEG: Average Improvement = 62.36%
Chorales: Average Improvement = 3.20%
Air_Quality: Average Improvement = 3.06%
"""

"""
Min values per horizon for air_quality:
  Horizon 5:
    Minimum test_mae: 0.4956748485565186 (Corresponding test_mse: 0.5138453245162964)
  Horizon 10:
    Minimum test_mae: 0.5792484879493713 (Corresponding test_mse: 0.6416391730308533)
  Horizon 15:
    Minimum test_mse: 0.7171288132667542 (Corresponding test_mae: 0.6267962455749512)
  Horizon 25:
    Minimum test_mse: 0.7795374989509583 (Corresponding test_mae: 0.6500051021575928)
Min values per horizon for chorales:
  Horizon 1:
    Minimum test_mse: 0.2234929651021957 (Corresponding test_mae: 0.1992906779050827)
  Horizon 2:
    Minimum test_mse: 0.2422483265399933 (Corresponding test_mae: 0.2047905921936035)
  Horizon 3:
    Minimum test_mse: 0.2595462501049042 (Corresponding test_mae: 0.2256415337324142)
  Horizon 4:
    Minimum test_mse: 0.2601789534091949 (Corresponding test_mae: 0.2264223992824554)ֿ
    
Min values per horizon for ettm1:
  Horizon 48:
    Minimum test_mse: 0.3062384724617004 (Corresponding test_mae: 0.3582214713096619)
  Horizon 96:
    Minimum test_mse: 0.2412273436784744 (Corresponding test_mae: 0.3073456287384033)
  Horizon 192:
    Minimum test_mse: 0.3405168056488037 (Corresponding test_mae: 0.3787482678890228)
  Horizon 336:
    Minimum test_mse: 0.4128028452396393 (Corresponding test_mae: 0.4390197992324829)
  
Min values per horizon for traffic:
  Horizon 1:
    Minimum test_mse: 0.2397979646921158 (Corresponding test_mae: 0.200151264667511)
  Horizon 16:
    Minimum test_mse: 0.4528388679027557 (Corresponding test_mae: 0.3007679283618927)
  Horizon 32:
    Minimum test_mse: 0.5488532185554504 (Corresponding test_mae: 0.3562396168708801)
  Horizon 48:
    Minimum test_mse: 0.6611170172691345 (Corresponding test_mae: 0.410185307264328)

Min values per horizon for electricity:
  Horizon 1:
    Minimum test_mse: 0.05179140716791153 (Corresponding test_mae: 0.1466573178768158)
  Horizon 16:
    Minimum test_mse: 0.110563613474369 (Corresponding test_mae: 0.2194186002016068)
  Horizon 32:
    Minimum test_mse: 0.1418711245059967 (Corresponding test_mae: 0.2562781572341919)

Min values per horizon for exchange:
  Horizon 96:
    Minimum test_mse: 0.06505617499351501 (Corresponding test_mae: 0.1739738136529922)
  Horizon 192:
    Minimum test_mse: 0.1640825420618057 (Corresponding test_mae: 0.294850617647171)
  Horizon 336:
    Minimum test_mse: 0.4229957461357117 (Corresponding test_mae: 0.4785425066947937)
  Horizon 720:
    Minimum test_mse: 0.7732616662979126 (Corresponding test_mae: 0.6843751072883606)
Aggregated Min values per horizon for noaa:
  Horizon 15:
    Minimum test_mse: 0.3013424277305603 (Corresponding test_mae: 0.4359403550624847)
  Horizon 30:
    Minimum test_mse: 0.3130705654621124 (Corresponding test_mae: 0.4477791488170624)
  Horizon 60:
    Minimum test_mse: 0.325585812330246 (Corresponding test_mae: 0.4576222896575928)
  Horizon 90:
    Minimum test_mse: 0.3306412994861603 (Corresponding test_mae: 0.4592036008834839)
Aggregated Min values per horizon for ecg_single:
  Horizon 1:
    Minimum test_mse: 0.08033280074596405 (Corresponding test_mae: 0.2043430507183075)
  Horizon 10:
    Minimum test_mse: 0.172207698225975 (Corresponding test_mae: 0.305258572101593)
  Horizon 25:
    Minimum test_mse: 0.2584512233734131 (Corresponding test_mae: 0.3754371106624603)
  Horizon 100:
    Minimum test_mse: 0.3936669826507568 (Corresponding test_mae: 0.4811891317367554)
Aggregated Min values per horizon for eeg_single:
  Horizon 1:
    Minimum test_mse: 0.07534848153591156 (Corresponding test_mae: 0.1016187220811844)
  Horizon 10:
    Minimum test_mse: 0.1599802374839783 (Corresponding test_mae: 0.1943245679140091)
  Horizon 25:
    Minimum test_mse: 0.2380640506744385 (Corresponding test_mae: 0.2603403031826019)
"""