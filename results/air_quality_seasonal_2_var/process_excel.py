import pandas as pd
import os
model = 'nft'
for model in ['nft', 'DLinear', 'PatchTST', 'TimesNet']: #
    # Specify the directory containing the Excel files
    directory = f"/home/../multiTS/NFT/results/air_quality_seasonal_2_var/{model}"

    # Create an empty list to store DataFrames
    dfs = []
    # Iterate through all Excel files in the directory
    for file in os.listdir(directory):
        if file.endswith(".xlsx") or file.endswith(".xls"):  # Check for Excel files
            file_path = os.path.join(directory, file)
            # Read the Excel file into a DataFrame
            df = pd.read_excel(file_path)
            df['Data'] = df['Data'].astype(str).str[-8:-6]
            df = df[['Data', 'test_mse']]
            dfs.append(df)

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated DataFrame to a new Excel file
    output_file = os.path.join(directory, f"{model}_concatenated_file.xlsx")
    concatenated_df.to_excel(output_file, index=False)


    print(f"All files have been concatenated and saved to {output_file}")
