import numpy as np # type: ignore
from scipy import signal # type: ignore
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt # type: ignore
import wfdb # type: ignore
import pywt
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def mean_absolute_value(signal):
    return np.mean(np.abs(signal))

def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def zero_crossing(signal):
    return np.count_nonzero(np.diff(np.sign(signal)))

def slope_sign_change(signal):
    slope_signs = np.sign(np.diff(signal))
    return np.count_nonzero(np.diff(slope_signs))

def root_mean_square(signal):
    return np.sqrt(np.mean(np.square(signal)))

def read_file_data(path):
    record = wfdb.rdrecord(path)
    signals = record.p_signal
    sampling_frequency = record.fs
    channel_names = record.sig_name
    units = record.units
    return sampling_frequency, signals[:, 0]

def generate_data(n_samples=10240, sampling_rate=2048):
    time = np.arange(n_samples) / sampling_rate
    emg_signal = 0.5 * np.sin(2 * np.pi * 50 * time) 
    emg_signal += 0.25 * np.sin(2 * np.pi * 100 * time) 
    emg_signal += 0.75 * np.sin(2 * np.pi * 150 * time) 
    emg_signal += 0.8 * np.sin(2 * np.pi * 200 * time) 
    noise = np.random.normal(0, 0.2, size=n_samples)
    return emg_signal + noise

def band_pass_filter(emg_signal, sampling_rate=2048, lowcut=10, highcut=450):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(6, [low, high], btype='bandpass')
    return signal.filtfilt(b, a, emg_signal)

def get_psd(emg_signal, sampling_frequency):
    return signal.welch(emg_signal, sampling_frequency, nperseg=1024)

def extract_features(emg_signal):
    mav = mean_absolute_value(emg_signal)
    wl = waveform_length(emg_signal)
    zc = zero_crossing(emg_signal)
    ssc = slope_sign_change(emg_signal)
    rms = root_mean_square(emg_signal)
    return [mav, wl, zc, ssc, rms]

def summarize_wavelet_coeffs(wavelet_coeffs):
    summary = np.empty((len(wavelet_coeffs), 4))
    for i, coeffs in enumerate(wavelet_coeffs):
        summary[i, :] = np.array([np.mean(coeffs), np.std(coeffs), skew(coeffs), kurtosis(coeffs)])
    return summary

def get_all_features(emg_signal, sampling_frequency):
    band_passed_emg = band_pass_filter(emg_signal, sampling_frequency)
    # f_filtered, Pxx_den_filtered = get_psd(band_passed_emg, sampling_frequency)
    features = extract_features(band_passed_emg)
    wavelet_transform = pywt.wavedec(band_passed_emg, "bior3.3", level=4)
    wavelet_coeffs = summarize_wavelet_coeffs(wavelet_transform).flatten()
    # feature_names = ["Mean Absolute Value", "Waveform Length", "Zero Crossing Rate", "Slope Sign Change", "Root Mean Squared"] #ewl, emav
    return features # np.append(features, wavelet_coeffs)

def create_data_set(root):
    df = pd.DataFrame(columns=["Mean Absolute Value", "Waveform Length", "Zero Crossing Rate", "Slope Sign Change",
                                "Root Mean Squared",
                                # "AbsoluteCoeff1", "AbsoluteCoeff2", "AbsoluteCoeff3", "AbsoluteCoeff4",
                                # "Detail4_Mean", "Detail4_Std", "Detail4_Skew", "Detail4_Kurtosis", 
                                # "Detail3_Mean", "Detail3_Std", "Detail3_Skew", "Detail3_Kurtosis",
                                # "Detail2_Mean", "Detail2_Std", "Detail2_Skew", "Detail2_Kurtosis",
                                # "Detail1_Mean", "Detail1_Std", "Detail1_Skew", "Detail1_Kurtosis", 
                                "Gesture"])
    for file in os.listdir(root):
        if file.endswith(".hea"):
            filename = file[:-4]
            gesture = file[file.find("gesture")+7:file.find("gesture")+9]
            sampling_frequency, emg_data = read_file_data(f"{root}/{filename}")
            features = get_all_features(emg_data, sampling_frequency)
            data = np.append(features, gesture)
            df.loc[len(df)] = data
    return df

def test_classifier(data):
    X, y = data.drop(["Gesture"], axis=1), data["Gesture"]
    # print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classif  = KNeighborsClassifier(n_neighbors=3)
    classif.fit(X_train, y_train)
    y_preds = classif.predict(X_test)
    print(accuracy_score(y_test, y_preds))
    ConfusionMatrixDisplay.from_predictions(y_test, y_preds)
    plt.show()

def test_cluster(data):
    X, y = data.drop(["Gesture"], axis=1), data["Gesture"]
    clustering = AgglomerativeClustering(n_clusters=3).fit(X)
    y_preds = clustering.labels_
    print(list(y), y_preds)
    # print(accuracy_score(y, y_preds))
    # ConfusionMatrixDisplay.from_predictions(y, y_preds)
    # plt.show()

def visualize(path):
    emg_data = sampling_frequency, emg_data = read_file_data(path)
    f, Pxx_den = get_psd(emg_data, sampling_frequency)

    band_passed_emg = band_pass_filter(emg_data, sampling_frequency)
    f_filtered, Pxx_den_filtered = get_psd(band_passed_emg, sampling_frequency)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # # Plot data on each subplot
    axes[0, 0].plot(emg_data)
    axes[0, 0].set_title('Raw EMG Data')

    axes[0, 1].plot(band_passed_emg)
    axes[0, 1].set_title('Processed EMG Data')

    axes[1, 0].semilogy(f, Pxx_den)
    axes[1, 0].set_title('Orginal PSD')

    axes[1, 1].semilogy(f_filtered, Pxx_den_filtered )
    axes[1, 1].set_title('Filtered PSD')

    # # # Adjust layout to prevent overlapping
    plt.tight_layout()

    # # # Show the plot
    plt.show()

data = create_data_set("./temp_data")
test_classifier(data)
