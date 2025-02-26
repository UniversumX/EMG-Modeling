# EMG Signal Processing and Classification

## Overview
This project involves processing Electromyography (EMG) signals, extracting relevant features, and using machine learning techniques for classification and clustering. The code includes functions for reading EMG data, applying signal processing techniques, extracting statistical and wavelet-based features, and evaluating classification and clustering models.

## Features Implemented
- **Signal Processing**: Band-pass filtering, power spectral density (PSD) computation, and wavelet transform.
- **Feature Extraction**: Mean Absolute Value, Waveform Length, Zero Crossing Rate, Slope Sign Change, Root Mean Squared, and wavelet-based statistical features.
- **Machine Learning Models**:
  - k-Nearest Neighbors (k-NN) classifier for gesture recognition.
  - Agglomerative clustering for unsupervised learning.
- **Data Visualization**: Plotting raw and processed EMG signals, PSD before and after filtering, and confusion matrix for classification results.

## Dependencies
The code requires the following Python libraries:
```sh
pip install -r requirements.txt
```

## File Structure
- `read_file_data(path)`: Reads EMG data from a file.
- `generate_data(n_samples, sampling_rate)`: Generates synthetic EMG signals.
- `band_pass_filter(emg_signal, sampling_rate, lowcut, highcut)`: Applies a band-pass filter to the EMG signal.
- `extract_features(emg_signal)`: Extracts statistical features from the EMG signal.
- `summarize_wavelet_coeffs(wavelet_coeffs)`: Computes summary statistics from wavelet decomposition coefficients.
- `get_all_features(emg_signal, sampling_frequency)`: Extracts both statistical and wavelet-based features.
- `create_data_set(root)`: Reads and processes all EMG files in a directory to create a dataset.
- `test_classifier(data)`: Trains and evaluates a k-NN classifier.
- `test_cluster(data)`: Applies agglomerative clustering to the dataset.
- `visualize(path)`: Plots raw and processed EMG signals and their PSD.

## Usage
1. **Prepare Data**: Ensure EMG data files are stored in `./temp_data`.
2. **Run the Script**: Execute the script to process data and test classification.
```sh
python exploratory.py
```

3. **Results**: The classifier's accuracy and confusion matrix will be displayed.