
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_variables=5
n_timesteps=1000

def create_seasonality_series(n_variables=n_variables, n_timesteps=n_timesteps, seasonality_period=50, noise_level=0.1, save_data=False):
        
    # Create time vector
    time = np.arange(n_timesteps)

    # Generate synthetic multivariate time series data
    data = np.zeros((n_timesteps, n_variables))
    for i in range(n_variables):
        # Add high seasonality with different phases and amplitudes
        amplitude = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2 * np.pi)
        data[:, i] = amplitude * np.sin(2 * np.pi * time / seasonality_period + phase)

        # Add some noise
        data[:, i] += np.random.normal(0, noise_level, n_timesteps)

    # Convert to a pandas DataFrame for visualization and manipulation
    columns = [f"Variable_{i+1}" for i in range(n_variables)]
    synthetic_data = pd.DataFrame(data, columns=columns)

    # Plot the generated time series data
    synthetic_data.plot(figsize=(12, 6))
    plt.title("Synthetic Multivariate Time Series with High Seasonality")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.grid()
    plt.show()

    if save_data:
        directory = '/home/noam.koren/multiTS/NFT/data/seasonality/'
        file_path = os.path.join(directory, 'seasonality.pkl')
        os.makedirs(directory, exist_ok=True)
        synthetic_data.to_pickle(file_path)

def save_to_excel(df, file_path):
    # Ensure the file path ends with .xlsx
    if not file_path.endswith('.xlsx'):
        file_path += '.xlsx'
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Creating a new file instead.")
            combined_df = df
    else:
        combined_df = df
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        combined_df.to_excel(writer, index=False)

    print(f"Data successfully saved to {file_path}")

def process_component_info(seasonality_trend_info, series_name):
    component_info = pd.DataFrame(seasonality_trend_info)
    component_info['Series_Name'] = series_name

    averages = component_info.select_dtypes(include='number').mean()
    averages_row = {col: averages[col] if col in averages.index else None for col in component_info.columns}
    averages_row['Series_Name'] = series_name
    averages_row['Series'] = 'Average'
    averages_df = pd.DataFrame([averages_row])

    component_info = pd.concat([component_info, averages_df], ignore_index=True)

    cols = ['Series_Name'] + [col for col in component_info.columns if col != 'Series_Name']
    component_info = component_info[cols]

    print("Component Info with Averages:")
    print(component_info.head(6))

    return component_info

def create_time_series(
    num_series, 
    num_points,
    trend_amplitude=0.5, 
    noise_level=1, 
    seasonal_amplitude=1, # (0.5, 1.5)
    save_data=False,
    series_name=None,
    compute_components=True  # parameter to compute seasonality and trend proportions
):
    time = np.arange(num_points)
    data = {}
    seasonality_trend_info = []

    for i in range(num_series):
        trend = trend_amplitude * (time / num_points)
        
        phase = np.random.uniform(0, 2 * np.pi)
        seasonal = (seasonal_amplitude * (i+1)) * np.sin(2 * np.pi * time / (50 + i * 5) + i * phase)
    
        noise = np.random.normal(0, noise_level, num_points)
        
        series = trend + seasonal + noise
        data[f"Series_{i+1}"] = series
        
        if compute_components:
            total_variance = np.var(series)
            seasonal_variance = np.var(seasonal)
            trend_variance = np.var(trend)
            noise_variance = np.var(noise)
        

            seasonality_ratio = seasonal_variance / total_variance
            trend_ratio = trend_variance / total_variance
            trend_to_seasonality_ratio = trend_variance / seasonal_variance if seasonal_variance > 0 else np.inf
            seasonality_to_trend_ratio = seasonal_variance / trend_variance if trend_variance > 0 else np.inf
            noise_ratio = noise_variance / total_variance
            
            trend_dominance = max(0, 1 - (noise_variance / (noise_variance + trend_variance)))
            seasonality_dominance = max(0, 1 - (noise_variance / (noise_variance + seasonal_variance)))

            seasonality_trend_info.append({
                'Series': f"Series_{i+1}",
                'Seasonality_Dominance': seasonality_dominance,
                'Trend_Dominance': trend_dominance,
                'Seasonality_Ratio': seasonality_ratio,
                'Trend_Ratio': trend_ratio,
                'Trend_to_Seasonality_Ratio': trend_to_seasonality_ratio,
                'Seasonality_to_Trend_Ratio': seasonality_to_trend_ratio,
                'Noise_Ratio': noise_ratio
            })

    synthetic_data = pd.DataFrame(data, index=time)
    # synthetic_data.plot(figsize=(10, 6), title="Synthetic Multivariate Time Series")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.show()
    
    component_info = process_component_info(seasonality_trend_info, series_name=series_name)

    
    if save_data:
        directory = f'/home/noam.koren/multiTS/NFT/data/{series_name}/'
        file_path = os.path.join(directory, f'{series_name}.pkl')
        os.makedirs(directory, exist_ok=True)
        synthetic_data.to_pickle(file_path)
        
        save_to_excel(component_info, '/home/noam.koren/multiTS/NFT/models/tests/analyse_data/data_components.xlsx')
    
for seasonal_amplitude in [0.01, 0.05, 0.1, 0.15, 0.2]:
    for trend_amplitude in [30, 40, 50, 60]:
        create_time_series(
            num_series=n_variables, 
            num_points=n_timesteps,
            trend_amplitude=trend_amplitude, 
            noise_level=0.1, 
            seasonal_amplitude=seasonal_amplitude,
            save_data=True,
            series_name=f'seasonal_{seasonal_amplitude}_trend_{trend_amplitude}'
            )