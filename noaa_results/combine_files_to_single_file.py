import os
import pandas as pd

# Directory containing the subdirectories with Excel files
root_dir = '/home/noam.koren/multiTS/NFT/results'

# List to store data from each Excel file
all_data = []

# Traverse through the directories and subdirectories
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Check if the file is an Excel file
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(subdir, file)
            # Read the Excel file
            df = pd.read_excel(file_path)
            # Append the data to the list
            all_data.append(df)

# Combine all data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save the combined data to a new Excel file
output_file_path = os.path.join(root_dir, 'combined_data.xlsx')
combined_data.to_excel(output_file_path, index=False)

print(f"All data combined into {output_file_path}")
