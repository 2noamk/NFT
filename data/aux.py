import pickle
import pandas as pd
import numpy as np

def pkl_to_csv(pkl_file, csv_file):
    try:
        # Load the data from the .pkl file
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)[0,:,:]

        # Handle different data formats
        if isinstance(data, pd.DataFrame):
            # If it's already a DataFrame, save it as a CSV
            data.to_csv(csv_file, index=False)
            print(f"Successfully converted {pkl_file} to {csv_file}.")
        elif isinstance(data, np.ndarray):
            # If it's a NumPy array, convert to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            print(f"Successfully converted {pkl_file} to {csv_file}.")
        elif isinstance(data, list):
            # If it's a list, attempt to convert to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            print(f"Successfully converted {pkl_file} to {csv_file}.")
        elif isinstance(data, dict):
            # If it's a dictionary, attempt to convert to DataFrame
            df = pd.DataFrame.from_dict(data, orient='index')
            df.to_csv(csv_file, index=False)
            print(f"Successfully converted {pkl_file} to {csv_file}.")
        else:
            print(f"Unsupported data format in {pkl_file}. Please check the data structure.")
    except Exception as e:
        print(f"Error converting {pkl_file} to CSV: {e}")

# Example usage
pkl_file = "/home/noam.koren/multiTS/NFT/data/illness/illness_36l_12h_0label/test_X.pkl"  # Replace with your .pkl file path
csv_file = "/home/noam.koren/multiTS/NFT/data/illness/illness_36l_12h_0label/test_X.csv"  # Replace with your .csv file path
pkl_to_csv(pkl_file, csv_file)



# import pickle
# import numpy as np

# def are_pickles_identical(file1, file2):
#     try:
#         # Load the contents of both pickle files
#         with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
#             data1 = pickle.load(f1)
#             data2 = pickle.load(f2)
        
#         # Handle cases where data are NumPy arrays
#         if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
#             return np.array_equal(data1, data2)
        
#         # Handle cases where data are lists of NumPy arrays or similar structures
#         if isinstance(data1, list) and isinstance(data2, list):
#             if len(data1) != len(data2):
#                 return False
#             return all(
#                 np.array_equal(d1, d2) if isinstance(d1, np.ndarray) else d1 == d2
#                 for d1, d2 in zip(data1, data2)
#             )
        
#         # Default to standard equality for other data types
#         return data1 == data2
#     except Exception as e:
#         print(f"Error comparing pickle files: {e}")
#         return False
# # Example usage
# file1 = "/home/noam.koren/multiTS/NFT/data/exchange_rate/test_X.pkl"  # Replace with your file path
# file2 = "/home/noam.koren/multiTS/NFT/data/exchange/test_X.pkl"  # Replace with your file path

# # print(are_pickles_identical(file1, file2))

# if are_pickles_identical(file1, file2):
#     print("The pickle files are identical.")
# else:
#     print("The pickle files are not identical.")
