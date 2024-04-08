import pandas as pd
import numpy as np

# Load the data
df = pd.read_excel('/home/noam.koren/multiTS/NFT/results/combined_data.xlsx')  # Replace with your actual file name

# Group the data and calculate the averages
group_columns = ['Lookback', 'Horizon', 'Epochs', 'Blocks', 'Is Poly']
numeric_columns = ['train_mse', 'test_mse', 'train_smape', 'test_smape', 'train_mape', 'test_mape', 'train_mase', 'test_mase']

# Replace empty cells and inf with NaN
df[numeric_columns] = df[numeric_columns].replace(['', np.inf, -np.inf], np.nan)

# Convert to floats, in case they are not already
df[numeric_columns] = df[numeric_columns].astype(float)

# Calculate the mean, ignoring NaN values
averaged_df = df.groupby(group_columns, as_index=False)[numeric_columns].mean()

# Save the averaged data to a new Excel file
averaged_file_path = '/home/noam.koren/multiTS/NFT/results/averaged_data.xlsx'  # Change the file name if needed
averaged_df.to_excel(averaged_file_path, index=False)
