import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    averages_row['Seasonality_Precetage'] = averages_row['Seasonality_Dominance'] / (averages_row['Seasonality_Dominance']  + averages_row['Trend_Dominance'])
    averages_df = pd.DataFrame([averages_row])

    component_info = pd.concat([component_info, averages_df], ignore_index=True)

    cols = ['Series_Name'] + [col for col in component_info.columns if col != 'Series_Name']
    component_info = component_info[cols]

    print("Averaged Component Info:")
    print(averages_df[cols].head(6))

    return component_info, averages_df[cols]

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
        directory = f'/home/../multiTS/NFT/data/{series_name}/'
        file_path = os.path.join(directory, f'{series_name}.pkl')
        os.makedirs(directory, exist_ok=True)
        synthetic_data.to_pickle(file_path)
        
        save_to_excel(component_info, '/home/../multiTS/NFT/models/tests/analyse_data/data_components.xlsx')

def get_components(y):
    
    # Step 2: Apply Fast Fourier Transform (FFT)
    n = len(y)  # Length of the signal
    fft_vals = np.fft.fft(y)  # FFT
    frequencies = np.fft.fftfreq(n)  # Frequency bins

    # Step 3: Identify dominant frequencies (filtering)
    # Sort by the magnitude of the FFT values and retain top frequencies
    magnitude = np.abs(fft_vals)
    threshold = 0.05 * max(magnitude)  # Keep frequencies with > 5% of max amplitude
    filtered_fft_vals = fft_vals.copy()
    filtered_fft_vals[magnitude < threshold] = 0  # Zero out low-amplitude components

    # Step 4: Reconstruct the seasonality using Inverse FFT (IFFT)
    seasonality = np.fft.ifft(filtered_fft_vals).real
    
    # Step 5: Filter out high frequencies (low-pass filter)
    # Define a cutoff frequency: retain only the low frequencies
    cutoff = 0.05  # Adjust this value based on your data (0 < cutoff < 1)
    filtered_fft_vals = fft_vals.copy()
    filtered_fft_vals[np.abs(frequencies) > cutoff] = 0  # Zero out high-frequency components

    # Step 4: Reconstruct the trend using Inverse FFT (IFFT)
    trend = np.fft.ifft(filtered_fft_vals).real
    
    noise = y - seasonality - trend
    
    return seasonality, trend, noise
 
def plot_componentes(y, seasonality, trend, noise):
    plt.figure(figsize=(12, 6))
    plt.plot(y, label='Original Series', color='blue')
    plt.plot(seasonality, label='Extracted Seasonality', color='red')
    plt.plot(trend, label='Extracted Trend', color='orange')
    plt.plot(noise, label='Extracted Noise', color='green')
    plt.title('Components Extraction using Fourier Transform')
    plt.legend()
    plt.show()
 
def calculate_components_dominance(data, data_name, save_excel=False, average=True):
    seasonality_trend_info = []
    for column in data.columns:
        y = data[column]
            
        seasonal, trend, noise = get_components(y)

        total_variance = y.var()
        seasonal_variance = seasonal.var()
        trend_variance = trend.var()
        noise_variance = noise.var()

        seasonality_dominance = max(0, 1 - (noise_variance / (noise_variance + seasonal_variance)))
        trend_dominance = max(0, 1 - (noise_variance / (noise_variance + trend_variance)))
        seasonality_precetage = seasonality_dominance / (seasonality_dominance + trend_dominance) if (seasonality_dominance + trend_dominance) != 0 else 0
        
        print(seasonality_dominance, trend_dominance, seasonality_precetage)
        
        seasonality_trend_info.append({
            'Series': column,
            'Seasonality_Dominance': seasonality_dominance,
            'Trend_Dominance': trend_dominance,
            'Seasonality_Precetage': seasonality_precetage,
            'total_variance': total_variance,
            'seasonal_variance': seasonal_variance,
            'trend_variance': trend_variance,
            'noise_variance': noise_variance,
            })
        
    component_info, averages_df = process_component_info(seasonality_trend_info, series_name=data_name)
    d = averages_df if average else component_info
    if save_excel:
        save_to_excel(d, f'/home/../multiTS/NFT/models/tests/analyse_data/{data_name}_components.xlsx')

    return component_info, averages_df



data = pd.read_pickle('/home/../multiTS/NFT/data/air_quality/air_quality.pkl')
component_info, averages_df = calculate_components_dominance(data, data_name='air_quality', save_excel=True, average=False)
