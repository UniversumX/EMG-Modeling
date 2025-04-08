import numpy as np # type: ignore
from scipy import signal # type: ignore
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt # type: ignore
import wfdb # type: ignore
import pywt
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans
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

def band_pass_filter(emg_signal, sampling_rate=2048, lowcut=10, highcut=200):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    # print(nyquist, low, high)
    # b, a = signal.butter(6, [low, high], btype='bandpass')
    b, a = signal.butter(4, [low, high], btype='bandpass', analog=True)
    filtered = signal.filtfilt(b, a, emg_signal)
    threshold = 300
    clipped_points = (filtered >= threshold) | (filtered <= -threshold)
    filtered[clipped_points] = np.interp(np.flatnonzero(clipped_points), np.flatnonzero(~clipped_points), filtered[~clipped_points])
    return filtered

def get_psd(emg_signal, sampling_frequency):
    return signal.welch(emg_signal, sampling_frequency, nperseg=1024)

def extract_features(emg_signal):
    # print(emg_signal.shape)
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
    # band_passed_emg = band_pass_filter(emg_signal, sampling_frequency)
    # f_filtered, Pxx_den_filtered = get_psd(band_passed_emg, sampling_frequency)
    features = extract_features(emg_signal)
    # wavelet_transform = pywt.wavedec(band_passed_emg, "bior3.3", level=4)
    # wavelet_coeffs = summarize_wavelet_coeffs(wavelet_transform).flatten()
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
    # # print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classif  = KNeighborsClassifier(n_neighbors=3)
    classif.fit(X_train, y_train)
    y_preds = classif.predict(X_test)
    # print(accuracy_score(y_test, y_preds))
    ConfusionMatrixDisplay.from_predictions(y_test, y_preds)
    plt.show()

def test_cluster(X):
    # X, y = data.drop(["Gesture"], axis=1), data["Gesture"]
    clustering = KMeans(n_clusters=3).fit(X)
    y_preds = clustering.labels_
    return y_preds
    # print(list(y), y_preds)
    # # print(accuracy_score(y, y_preds))
    # ConfusionMatrixDisplay.from_predictions(y, y_preds)
    # plt.show()

def visualize(path):
    sampling_frequency, emg_data = read_file_data(path)
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

def csv_visualize():
    tricep1 = np.array(pd.read_csv("data/emg_data_viraj_left_tricep_lower.csv", header = 0, index_col = 0)["TRICEP"])
    bicep1 = np.array(pd.read_csv("data/emg_data_viraj_left_tricep_lower.csv", header = 0, index_col = 0)["BICEP"])


    tricep2 = np.array(pd.read_csv("data/emg_data_viraj_left.csv", header = 0, index_col = 0)["TRICEP"])
    bicep2 = np.array(pd.read_csv("data/emg_data_viraj_left.csv", header = 0, index_col = 0)["BICEP"])


    tricep3 = np.array(pd.read_csv("data/emg_data_viraj_right_tricep_lower.csv", header = 0, index_col = 0)["TRICEP"])
    bicep3 = np.array(pd.read_csv("data/emg_data_viraj_right_tricep_lower.csv", header = 0, index_col = 0)["BICEP"])


    tricep4 = np.array(pd.read_csv("data/emg_data_viraj_right.csv", header = 0, index_col = 0)["TRICEP"])
    bicep4 = np.array(pd.read_csv("data/emg_data_viraj_right.csv", header = 0, index_col = 0)["BICEP"])

    
    # sampling_frequency, emg_data = read_file_data(path)

    sampling_frequency = 1000

    # print(np.max(bicep), np.percentile(bicep, 85)) # 2645.0 1824.0 1182.0 2040.0 2361.0
    
    bicep_emg_data = bicep1
    tricep_emg_data = tricep1
    # bicep_emg_data[bicep < np.percentile(bicep, 65)] = 0 
    bicep_emg_data[bicep1 < 1000] = 0 
    tricep_emg_data[tricep1 < 275] = 0 

    f_bicep, Pxx_den_bicep = get_psd(bicep_emg_data, sampling_frequency)

    # tricep_emg_data = tricep
    # f_tricep, Pxx_den_tricep = get_psd(tricep_emg_data, sampling_frequency)
    # band_passed_tricep_emg = band_pass_filter(tricep_emg_data, sampling_frequency)

    low_pass_emg = band_pass_filter(bicep_emg_data, sampling_frequency, lowcut=10, highcut=150)
    f_filtered_low, Pxx_den_filtered_low = get_psd(low_pass_emg, sampling_frequency)

    mid_pass_emg = band_pass_filter(bicep_emg_data, sampling_frequency, lowcut=150, highcut=300)

    high_pass_emg = band_pass_filter(bicep_emg_data, sampling_frequency, lowcut=300, highcut=500)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # # Plot data on each subplot
    axes[0, 0].plot(bicep_emg_data)
    axes[0, 0].plot(tricep_emg_data)
    axes[0, 0].set_title('Raw EMG Data')

    bicep_emg_data = bicep2.copy()
    tricep_emg_data = tricep2.copy()
    # bicep_emg_data[bicep < np.percentile(bicep, 65)] = 0 
    bicep_emg_data[(bicep2 < 1000) | (bicep2 > 2750)] = 0 
    bicep_emg_extra = bicep2.copy()
    bicep_emg_extra[(bicep2 < 1000) | (bicep2 > 4000)] = 0 
    # bicep_emg_data[bicep2 < 1000] = 0 
    tricep_emg_data[tricep2 < 275] = 0 

    # # Plot data on each subplot
    axes[0, 1].plot(bicep_emg_extra)
    axes[0, 1].plot(bicep_emg_data)
    axes[0, 1].plot(tricep_emg_data)
    axes[0, 1].set_title('Raw EMG Data')

    bicep_emg_data = bicep3
    tricep_emg_data = tricep3
    # bicep_emg_data[bicep < np.percentile(bicep, 65)] = 0 
    bicep_emg_data[bicep3 < 1000] = 0 
    tricep_emg_data[tricep3 < 275] = 0 

    # # Plot data on each subplot
    axes[1, 0].plot(bicep_emg_data)
    axes[1, 0].plot(tricep_emg_data)
    axes[1, 0].set_title('Raw EMG Data')

    bicep_emg_data = bicep4
    tricep_emg_data = tricep4
    # bicep_emg_data[bicep < np.percentile(bicep, 65)] = 0 
    bicep_emg_data[bicep4 < 1000] = 0 
    tricep_emg_data[tricep4 < 275] = 0 

    # # Plot data on each subplot
    axes[1, 1].plot(bicep_emg_data)
    axes[1, 1].plot(tricep_emg_data)
    axes[1, 1].set_title('Raw EMG Data')


    # axes[0, 1].plot(low_pass_emg)
    # axes[0, 1].set_title('Low Passed EMG Data (10-150)')

    # axes[1, 0].plot(mid_pass_emg)
    # axes[1, 0].set_title('Mid Passed EMG Data (150-300)')

    # axes[1, 1].semilogy(f_filtered, Pxx_den_filtered)
    # axes[1, 1].plot(high_pass_emg)
    # axes[1, 1].semilogy(f_bicep, Pxx_den_bicep)
    # axes[1, 1].set_title('High Passed EMG Data (300-500)')

    # # # Adjust layout to prevent overlapping
    plt.tight_layout()

    # # # Show the plot
    plt.show()

def create_batches(arr, batch_size=25):
    batches = [np.array(arr[i:i + batch_size]) for i in range(0, len(arr)-batch_size)]
    return batches

def old_data_from_df():
    df = pd.DataFrame(columns=["Low Mean Absolute Value", "Low Waveform Length", "Low Zero Crossing Rate", "Low Slope Sign Change",
                                "Low Root Mean Squared",
                                "Mid Mean Absolute Value", "Mid Waveform Length", "Mid Zero Crossing Rate", "Mid Slope Sign Change",
                                "Mid Root Mean Squared",
                                "High Mean Absolute Value", "High Waveform Length", "High Zero Crossing Rate", "High Slope Sign Change",
                                "High Root Mean Squared",
                                # "AbsoluteCoeff1", "AbsoluteCoeff2", "AbsoluteCoeff3", "AbsoluteCoeff4",
                                # "Detail4_Mean", "Detail4_Std", "Detail4_Skew", "Detail4_Kurtosis", 
                                # "Detail3_Mean", "Detail3_Std", "Detail3_Skew", "Detail3_Kurtosis",
                                # "Detail2_Mean", "Detail2_Std", "Detail2_Skew", "Detail2_Kurtosis",
                                # "Detail1_Mean", "Detail1_Std", "Detail1_Skew", "Detail1_Kurtosis", 
                                ])
    

    tricep = np.array(pd.read_csv("emg_data_new.csv", header = 0, index_col = 0)["TRICEP"])
    bicep = np.array(pd.read_csv("emg_data_new.csv", header = 0, index_col = 0)["BICEP"])

    sampling_frequency = 1000

    emg_data = np.array(bicep)

    low_pass_emg = band_pass_filter(emg_data, sampling_frequency, lowcut=10, highcut=150)
    mid_pass_emg = band_pass_filter(emg_data, sampling_frequency, lowcut=150, highcut=300)
    high_pass_emg = band_pass_filter(emg_data, sampling_frequency, lowcut=300, highcut=500)

    print("filters done")

    low_batches = create_batches(low_pass_emg, batch_size=25)   
    mid_batches = create_batches(mid_pass_emg, batch_size=25)   
    high_batches = create_batches(high_pass_emg, batch_size=25)   

    print("batching done")
    print(len(low_batches))

    # print(np.stack(low_batches).shape)

    for i in range(len(low_batches)): 
        low_features = get_all_features(low_batches[i], sampling_frequency)
        mid_features = get_all_features(mid_batches[i], sampling_frequency)
        high_features = get_all_features(high_batches[i], sampling_frequency)
        data = np.concatenate([low_features, mid_features, high_features])
        # print(len(data))
        df.loc[len(df)] = data
        if (i % 500 == 0):
            print(f"batch {i} done")

    print("dataframe created")
    df.to_csv("extracted_features.csv")
    return df

def cluster_visualize(signal, clusters):
    time = np.array([i for i in range(len(signal))])
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label='Signal', color='black', alpha=0.7)

    for cluster_id in np.unique(clusters):
        cluster_mask = (clusters == cluster_id)
        plt.scatter(time[cluster_mask], signal[cluster_mask], label=f'Cluster {cluster_id}', alpha=0.7)

    # Adding labels and title
    plt.title("Signal with Cluster Predictions")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Amplitude")
    plt.legend(loc='upper right')
    plt.show()

def create_data_set_from_dataframe():
    # Modified column names to include both bicep and tricep features
    df = pd.DataFrame(columns=[
        "Source File",  # New column for tracking data source
        # Bicep features
        "Bicep Low Mean Absolute Value", "Bicep Low Waveform Length", "Bicep Low Zero Crossing Rate", "Bicep Low Slope Sign Change",
        "Bicep Low Root Mean Squared",
        "Bicep Mid Mean Absolute Value", "Bicep Mid Waveform Length", "Bicep Mid Zero Crossing Rate", "Bicep Mid Slope Sign Change",
        "Bicep Mid Root Mean Squared",
        "Bicep High Mean Absolute Value", "Bicep High Waveform Length", "Bicep High Zero Crossing Rate", "Bicep High Slope Sign Change",
        "Bicep High Root Mean Squared",
        # Tricep features
        "Tricep Low Mean Absolute Value", "Tricep Low Waveform Length", "Tricep Low Zero Crossing Rate", "Tricep Low Slope Sign Change",
        "Tricep Low Root Mean Squared",
        "Tricep Mid Mean Absolute Value", "Tricep Mid Waveform Length", "Tricep Mid Zero Crossing Rate", "Tricep Mid Slope Sign Change",
        "Tricep Mid Root Mean Squared",
        "Tricep High Mean Absolute Value", "Tricep High Waveform Length", "Tricep High Zero Crossing Rate", "Tricep High Slope Sign Change",
        "Tricep High Root Mean Squared",
    ])
    
    # List of input files to process
    input_files = [
        "emg_data_viraj_left_tricep_lower.csv",
        "emg_data_viraj_left.csv",
        "emg_data_viraj_right_tricep_lower.csv",
        "emg_data_viraj_right.csv"
    ]

    sampling_frequency = 1000

    for file in input_files:
        print(f"Processing {file}")
        data = pd.read_csv(f"data/{file}", header=0, index_col=0)
        tricep = np.array(data["TRICEP"])
        bicep = np.array(data["BICEP"])

        # Process bicep data
        bicep_low = band_pass_filter(bicep, sampling_frequency, lowcut=10, highcut=150)
        bicep_mid = band_pass_filter(bicep, sampling_frequency, lowcut=150, highcut=300)
        bicep_high = band_pass_filter(bicep, sampling_frequency, lowcut=300, highcut=500)

        # Process tricep data
        tricep_low = band_pass_filter(tricep, sampling_frequency, lowcut=10, highcut=150)
        tricep_mid = band_pass_filter(tricep, sampling_frequency, lowcut=150, highcut=300)
        tricep_high = band_pass_filter(tricep, sampling_frequency, lowcut=300, highcut=500)

        # Create batches
        bicep_low_batches = create_batches(bicep_low)
        bicep_mid_batches = create_batches(bicep_mid)
        bicep_high_batches = create_batches(bicep_high)
        
        tricep_low_batches = create_batches(tricep_low)
        tricep_mid_batches = create_batches(tricep_mid)
        tricep_high_batches = create_batches(tricep_high)

        print(f"Processing {len(bicep_low_batches)} batches")

        for i in range(len(bicep_low_batches)):
            # Extract features for bicep
            bicep_low_features = get_all_features(bicep_low_batches[i], sampling_frequency)
            bicep_mid_features = get_all_features(bicep_mid_batches[i], sampling_frequency)
            bicep_high_features = get_all_features(bicep_high_batches[i], sampling_frequency)
            
            # Extract features for tricep
            tricep_low_features = get_all_features(tricep_low_batches[i], sampling_frequency)
            tricep_mid_features = get_all_features(tricep_mid_batches[i], sampling_frequency)
            tricep_high_features = get_all_features(tricep_high_batches[i], sampling_frequency)

            # Combine all features
            data = [file] + list(bicep_low_features) + list(bicep_mid_features) + list(bicep_high_features) + \
                   list(tricep_low_features) + list(tricep_mid_features) + list(tricep_high_features)
            
            df.loc[len(df)] = data

            if (i % 500 == 0):
                print(f"batch {i} done")

    print("dataframe created")
    df.to_csv("extracted_features_updated.csv")
    return df

def preds_visualize():
    # Read EMG data for all files
    tricep1 = np.array(pd.read_csv("data/emg_data_viraj_left_tricep_lower.csv", header=0, index_col=0)["TRICEP"])
    bicep1 = np.array(pd.read_csv("data/emg_data_viraj_left_tricep_lower.csv", header=0, index_col=0)["BICEP"])
    
    tricep2 = np.array(pd.read_csv("data/emg_data_viraj_left.csv", header=0, index_col=0)["TRICEP"]) 
    bicep2 = np.array(pd.read_csv("data/emg_data_viraj_left.csv", header=0, index_col=0)["BICEP"])
    
    tricep3 = np.array(pd.read_csv("data/emg_data_viraj_right_tricep_lower.csv", header=0, index_col=0)["TRICEP"])
    bicep3 = np.array(pd.read_csv("data/emg_data_viraj_right_tricep_lower.csv", header=0, index_col=0)["BICEP"])
    
    tricep4 = np.array(pd.read_csv("data/emg_data_viraj_right.csv", header=0, index_col=0)["TRICEP"])
    bicep4 = np.array(pd.read_csv("data/emg_data_viraj_right.csv", header=0, index_col=0)["BICEP"])

    # Read features data and get predictions for each file
    features_df = pd.read_csv("extracted_features_updated.csv", header=0, index_col=0)
    
    # File names to match with features
    file_names = ['emg_data_viraj_left_tricep_lower.csv', 
                  'emg_data_viraj_left.csv',
                  'emg_data_viraj_right_tricep_lower.csv', 
                  'emg_data_viraj_right.csv']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    titles = ['Left Tricep Lower', 'Left', 'Right Tricep Lower', 'Right']
    
    for i, (ax, bicep, tricep, title, file_name) in enumerate(zip(
            axes.flat, 
            [bicep1, bicep2, bicep3, bicep4], 
            [tricep1, tricep2, tricep3, tricep4], 
            titles,
            file_names)):
            
        # Get features for this file only
        file_features = features_df[features_df['Source File'] == file_name]
        
        # Combine bicep and tricep features
        combined_features = pd.concat([
            file_features[[col for col in file_features.columns if col.startswith('Bicep Low')]],
            file_features[[col for col in file_features.columns if col.startswith('Tricep Low')]]
        ], axis=1)
        
        # Get single set of predictions for both muscles
        preds = test_cluster(combined_features)

        print(file_name)
        print(np.unique(preds, return_counts=True))
        
        # Add 25 zeros at start of predictions to align with batches
        preds_display = np.insert(preds, 0, np.zeros(25))
        
        time = np.arange(len(bicep))
        
        # Plot base EMG signals
        ax.plot(time, bicep, color='black', alpha=0.3, label='Bicep Signal')
        ax.plot(time, tricep, color='gray', alpha=0.3, label='Tricep Signal')
        # ax.plot(time, bicep, color='red', label='Bicep Signal')
        # ax.plot(time, tricep, color='blue', label='Tricep Signal')
        
        # Plot clusters with different colors
        colors = ['green', 'pink', 'yellow']
        for cluster_id in np.unique(preds_display):
            cluster_mask = (preds_display == cluster_id)
            # Plot both bicep and tricep points for this cluster
            ax.scatter(time[cluster_mask], bicep[cluster_mask], 
                      color=colors[int(cluster_id)], 
                      alpha=0.5, 
                      s=10,
                      label=f'Cluster {int(cluster_id)} - Bicep')
            ax.scatter(time[cluster_mask], tricep[cluster_mask], 
                      color=colors[int(cluster_id)], 
                      alpha=0.5, 
                      s=10,
                      marker='s',  # square marker to distinguish from bicep
                      label=f'Cluster {int(cluster_id)} - Tricep')
                
        ax.set_title(f'EMG Data - {title}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()

def robert():
    robert_df = pd.read_csv("data/emg_data_robert_left.csv")
    robert_only_signal = robert_df.dropna(axis=0, how="any")
    bicep = robert_only_signal["BICEP"]
    tricep = robert_only_signal["TRICEP"]
    sampling_frequency = 4
    f_bicep, Pxx_den_bicep = get_psd(bicep, sampling_frequency)
    f_tricep, Pxx_den_tricep = get_psd(tricep, sampling_frequency)

    # band_passed_emg = band_pass_filter(emg_data, sampling_frequency)
    # f_filtered, Pxx_den_filtered = get_psd(band_passed_emg, sampling_frequency)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # # Plot data on each subplot
    axes[0, 0].plot(bicep)
    axes[0, 0].plot(tricep)
    axes[0, 0].set_title('Bicep EMG Data')

    # axes[0, 1].plot(band_passed_emg)
    # axes[0, 1].set_title('Processed EMG Data')

    axes[1, 0].semilogy(f_bicep, Pxx_den_bicep)
    axes[1, 0].semilogy(f_tricep, Pxx_den_tricep)
    axes[1, 0].set_title('Orginal PSD')

    # axes[1, 1].semilogy(f_filtered, Pxx_den_filtered )
    # axes[1, 1].set_title('Filtered PSD')

    # # # Adjust layout to prevent overlapping
    plt.tight_layout()

    # # # Show the plot
    plt.show()
    

robert()


# data = create_data_set("./temp_data")
# test_classifier(data)
# csv_visualize()
# data = create_data_set_from_dataframe()
# data = pd.read_csv("data/extracted_features.csv", header=0, index_col=0)
# data = data[[
#     # "Low Mean Absolute Value", "Low Waveform Length", "Low Zero Crossing Rate", "Low Slope Sign Change", "Low Root Mean Squared",
#     "Mid Mean Absolute Value", "Mid Waveform Length", "Mid Zero Crossing Rate", "Mid Slope Sign Change", "Mid Root Mean Squared",
#     # "High Mean Absolute Value", "High Waveform Length", "High Zero Crossing Rate", "High Slope Sign Change", "High Root Mean Squared",
#     ]]

# preds = (test_cluster(data))

# bicep = np.array(pd.read_csv("data/emg_data_new.csv", header = 0, index_col = 0)["BICEP"])

# sampling_frequency = 1000
# bicep = band_pass_filter(bicep, sampling_frequency, 150, 300)

# # print(np.shape(preds))
# print(np.shape(bicep))
# # print(np.shape(np.zeros(25)))
# preds_display = np.insert(preds, 0, np.zeros(25))
# print(np.shape(preds_display))
# print(np.unique(preds_display, return_counts=True))
# # cluster_visualize(bicep, preds_display)
# csv_visualize()

# Update the clustering visualization section to handle both muscles
# create_data_set_from_dataframe()
