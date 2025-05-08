import pandas as pd

def filter_rows(group):
    if len(group) == 1:
        return group
    else:
        sorted_group = group.sort_values(by=['test_smape', 'test_mase'])
        first, second = sorted_group.iloc[0], sorted_group.iloc[1]
        
        if first['test_smape'] < second['test_smape'] and first['test_mase'] < second['test_mase']:
            return first.to_frame().T
        elif first['test_smape'] > second['test_smape'] and first['test_mase'] > second['test_mase']:
            return second.to_frame().T
        else:
            return sorted_group

def count_is_poly(df, group_by_fields):
    return df.groupby(group_by_fields + ['Is Poly']).size().unstack(fill_value=0)

if __name__ == "__main__":  
    base_path = '/home/../multiTS/NFT/models/tests/test_trend/'

    df = pd.read_excel(base_path + 'evaluate_trend_syntatic_data.xlsx')
    
    filtered_df = df.groupby(['Lookback', 'Horizon', 'Vars', 'Epochs', 'Stacks', 'Dagree', 'Blocks']).apply(filter_rows).reset_index(drop=True)

    filtered_df.to_excel(base_path + 'filtered_data.xlsx', index=False)

    lookback_horizon_counts = count_is_poly(filtered_df, ['Lookback', 'Horizon'])
    lookback_horizon_counts.to_csv(base_path + 'lookback_horizon_counts.csv')

    vars_counts = count_is_poly(filtered_df, ['Vars'])
    vars_counts.to_csv(base_path + 'vars_counts.csv')

    blocks_counts = count_is_poly(filtered_df, ['Blocks'])
    blocks_counts.to_csv(base_path + 'blocks_counts.csv')
