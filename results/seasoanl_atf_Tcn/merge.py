import os
import pandas as pd

# Define the root directory containing the subdirectories
root_dir = "/home/noam.koren/multiTS/NFT/results/seasonal_itrans/"  # Replace with your actual directory path

# Initialize an empty list to collect data
data_list = []

# Walk through the directories
for seasonal_trend_dir in os.listdir(root_dir):
    full_path_seasonal_trend = os.path.join(root_dir, seasonal_trend_dir)
    if os.path.isdir(full_path_seasonal_trend):
        for model_dir in ["iTransformer"]:
            model_path = os.path.join(full_path_seasonal_trend, model_dir)
            if os.path.isdir(model_path):
                for file in os.listdir(model_path):
                    if file.endswith(".xlsx"):
                        file_path = os.path.join(model_path, file)
                        # Read the Excel file
                        try:
                            df = pd.read_excel(file_path)
                            # Extract relevant columns
                            for _, row in df.iterrows():
                                data_list.append({
                                    "Data": row.get("Data"),
                                    "Model": model_dir.lower(),
                                    "Test MSE": row.get("test_mse")
                                })
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

# Create a DataFrame from the collected data
output_df = pd.DataFrame(data_list)

# Save the consolidated DataFrame to a new Excel file
output_file_path = root_dir + "consolidated_results.xlsx"
output_df.to_excel(output_file_path, index=False)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Consolidated Results", dataframe=output_df)
