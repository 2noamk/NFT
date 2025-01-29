# Project Title
Neural Fourier Transform for Multiple Time Series Prediction

The Neural Fourier Transform (NFT) algorithm integrates multi-dimensional Fourier transforms with Temporal Convolutional Network layers to enhance forecasting accuracy and interpretability. The efficacy of the Neural Fourier Transform is empirically validated on six diverse datasets, demonstrating improvements over multiple forecasting horizons and lookbacks, thereby establishing new state-of-the-art results. Our contributions advance the field of time series forecasting by providing a model that not only excels in predictive performance but also in its capacity to provide interpretable results, a valuable asset for practitioners and researchers alike.

## Repository Structure

This repository contains the following directories:

- `data`: Contains directories for each dataset. Each dataset directory includes data files (`.pkl`) and scripts for data processing.
- `models`: Contains Python scripts for model training and implementation.
    - `NFT`:
      - Contains the `NFT.py` file which has the NFT model implemetantion.
        - Contains the `train_model.py` file which trains the NFT model. Run this script with the desired dataset.
    - `trained_models`: Directory where trained models are saved.
    - `baseline_models`: Contains the baseline models implemetantion.
- `results`: Contains directories for each dataset, and within those, directories for each model with training results.
    - Each model's directory includes results such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Symmetric Mean Absolute Percentage Error (sMAPE), Mean Absolute Percentage Error (MAPE), and Mean Absolute Scaled Error (MASE).
