import pandas as pd

base_path = '/home/../multiTS/NFT/models/tests/test_trend/'

df = pd.read_excel(f'{base_path}evaluate_trend_on_syntatic_data.xlsx') 

df['Is Poly'] = df['Is Poly'].apply(lambda x: True if str(x).lower() == 'true' else False)

df_poly_true = df[df['Is Poly'] == True]

lowest_mse = pd.DataFrame()
lowest_smape = pd.DataFrame()
lowest_mase = pd.DataFrame()

grouped = df_poly_true.groupby(['Lookback', 'Horizon', 'Vars', 'Epochs', 'Dagree', 'Blocks'])

for _, group in grouped:
    if not group.empty:
        min_mse_row = group.loc[group['test_mse'].idxmin()]
        lowest_mse = pd.concat([lowest_mse, min_mse_row.to_frame().T])

        min_smape_row = group.loc[group['test_smape'].idxmin()]
        lowest_smape = pd.concat([lowest_smape, min_smape_row.to_frame().T])

        min_mase_row = group.loc[group['test_mase'].idxmin()]
        lowest_mase = pd.concat([lowest_mase, min_mase_row.to_frame().T])

lowest_mse.drop_duplicates(inplace=True)
lowest_smape.drop_duplicates(inplace=True)
lowest_mase.drop_duplicates(inplace=True)

lowest_mse.to_excel(f'{base_path}lowest_mse.xlsx', index=False)
lowest_smape.to_excel(f'{base_path}lowest_smape.xlsx', index=False)
lowest_mase.to_excel(f'{base_path}lowest_mase.xlsx', index=False)

mse_counts = lowest_mse['Stacks'].value_counts()
smape_counts = lowest_smape['Stacks'].value_counts()
mase_counts = lowest_mase['Stacks'].value_counts()

with open(f'{base_path}stacks_summary_block_types.txt', 'w') as f:
    f.write("Counts of each 'Stacks' type in the DataFrame with the lowest MSE:\n")
    f.write(mse_counts.to_string())
    f.write("\n\nCounts of each 'Stacks' type in the DataFrame with the lowest SMAPE:\n")
    f.write(smape_counts.to_string())
    f.write("\n\nCounts of each 'Stacks' type in the DataFrame with the lowest MASE:\n")
    f.write(mase_counts.to_string())

