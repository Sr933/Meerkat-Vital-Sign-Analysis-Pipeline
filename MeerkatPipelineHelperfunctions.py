import os
import numpy as np
from scipy.signal import find_peaks
import simdkalman
import scipy
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns

# Camera calibration parameters
PX = 639.8630981445312
PY = 370.0270690917969
FX = 611.5502319335938
FY = 611.1410522460938


def butter_bandpass(lowcut, highcut, fs, order=7):
    # Create Butterworth bandpass filter of specified order and frequency cutoffs depending on sampling rate
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = scipy.signal.butter(
        order, [low, high], analog=False, btype="band", output="sos"
    )
    return sos


def PCA_decomposition(A):
    M = np.mean(A, axis=0)
    C = A - M
    V = np.cov(C)
    values, vectors = np.linalg.eig(V)
    return values, vectors


def X_to_TS(X_i):
    # Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series.
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    return np.array(
        [X_rev.diagonal(i).mean() for i in range(-X_i.shape[0] + 1, X_i.shape[1])]
    )


def PCA_timeseries_resp_signal(signal):
    ##PCA for time series using Hankel matrix after https://www.kaggle.com/code/leokaka/pca-for-time-series-analysis
    N = len(signal)
    L = 20  # The window length
    K = N - L + 1  # number of columns in the trajectory matrix
    # Create Hankel matrix
    X = np.column_stack([signal[i : i + L] for i in range(0, K)])
    d = 15
    values, vectors = PCA_decomposition(X)
    U_k = vectors
    X_elem_pca = np.array(
        [
            np.dot(
                np.dot(
                    np.expand_dims(U_k[:, i], axis=1),
                    np.expand_dims(U_k[:, i].T, axis=0),
                ),
                X,
            )
            for i in range(0, d)
        ]
    )
    # Recover timeseries from recalculated matrix
    F_trend = X_to_TS(X_elem_pca[[0, 5]].sum(axis=0))
    return F_trend


def PCA_timeseries(signal):
    # Bandpass filter

    N = len(signal)

    L = 20  # The window length
    K = N - L + 1  # number of columns in the trajectory matrix

    # Create matrix and perform PCA
    X = np.column_stack([signal[i : i + L] for i in range(0, K)])
    estimator = decomposition.PCA(n_components=1, svd_solver="full")
    X_elem_pca = estimator.inverse_transform(estimator.fit_transform(X))
    # Recover signal from reconstrucred matrix
    F_trend = X_to_TS(X_elem_pca)
    return F_trend


def PCA_respiratory_signal(
    average_depth,
    ROI_x_1_array,
    ROI_x_2_array,
    ROI_y_1_array,
    ROI_y_2_array,
    outliers=True,
):
    butter = butter_bandpass(15 / 60, 150 / 60, 30, order=7)
    filtered_data = scipy.signal.sosfiltfilt(butter, average_depth)
    F_trend = PCA_timeseries(filtered_data)

    # Calculate X and Y position in world coordinates for region of interest
    ROI_x1 = (
        np.multiply((np.array(ROI_x_1_array) - PX), F_trend + np.mean(average_depth))
        / FX
    )
    ROI_x2 = (
        np.multiply((np.array(ROI_x_2_array) - PX), F_trend + np.mean(average_depth))
        / FX
    )
    ROI_y1 = (
        np.multiply((np.array(ROI_y_1_array) - PY), F_trend + np.mean(average_depth))
        / FY
    )
    ROI_y2 = (
        np.multiply((np.array(ROI_y_2_array) - PY), F_trend + np.mean(average_depth))
        / FY
    )
    # Filter roi positions to reduce noise
    ROI_x1 = scipy.signal.sosfiltfilt(butter, ROI_x1) + np.mean(ROI_x1)
    ROI_x2 = scipy.signal.sosfiltfilt(butter, ROI_x2) + np.mean(ROI_x2)
    ROI_y1 = scipy.signal.sosfiltfilt(butter, ROI_y1) + np.mean(ROI_y1)
    ROI_y2 = scipy.signal.sosfiltfilt(butter, ROI_y2) + np.mean(ROI_y2)

    # Calculate volume signal
    chest_area = np.multiply((ROI_x2 - ROI_x1), (ROI_y2 - ROI_y1))
    F_trend = np.multiply(F_trend, chest_area) / 10**3 #convert units from mm^3

    # Assign 10000 to outlier values to remove them in later processing
    threshold=8
    mask = np.abs(F_trend) > threshold
    # Apply the condition and update F_trend
    F_trend[mask] = 10000 if outliers else 8
    return F_trend


def choose_subject(data_analysis_folder):
    """
    Prompts the user to select a subject from the files in the specified folder.
    
    Args:
        data_analysis_folder (str): The path to the folder containing subject files.
        
    Returns:
        str: The path to the selected subject folder.
    """
    # List all files in the directory and filter for files containing 'mk'
    filenames = [os.fsdecode(file) for file in os.listdir(data_analysis_folder) if "mk" in file]
    
    # Print file options
    for i, file in enumerate(filenames, start=1):
        print(f"{i} : {file}")
    
    # Prompt user for selection
    while True:
        try:
            user_choice = int(input("Desired subject: ")) - 1
            if 0 <= user_choice < len(filenames):
                file_folder = filenames[user_choice]
                break
            else:
                print("Out of range. Please enter a valid number.")
        except ValueError:
            print("Input must be a number. Please try again.")
    # Construct and return the full path to the selected subject folder
    subject_folder = os.path.join(data_analysis_folder, file_folder)
    return subject_folder



def find_ac_signal(filtered_signal, interval):
    peaks_signal, _ = find_peaks(filtered_signal, distance=6)
    valleys_signal, _ = find_peaks(-filtered_signal, distance=6)

    num_intervals = int((len(filtered_signal) - interval) / 30)

    signal_ac_signal = np.zeros(num_intervals)
    signal_dc_signal = np.zeros(num_intervals)

    for time in range(num_intervals):
        start_idx = 30 * time
        end_idx = interval + start_idx

        # Get indices of peaks and valleys within the current interval
        result_signal_peaks = np.where((peaks_signal >= start_idx) & (peaks_signal <= end_idx))[0]
        result_signal_valleys = np.where((valleys_signal >= start_idx) & (valleys_signal <= end_idx))[0]

        # Extract peak and valley values
        peaks = filtered_signal[peaks_signal[result_signal_peaks]]
        valleys = filtered_signal[valleys_signal[result_signal_valleys]]

        # Calculate amplitudes
        min_len = min(len(peaks), len(valleys))
        amplitudes = peaks[:min_len] - valleys[:min_len]

        # Calculate and store AC and DC signals
        signal_ac_signal[time] = np.median(amplitudes)
        signal_dc_signal[time] = np.median(filtered_signal[peaks_signal[result_signal_peaks[0]] : peaks_signal[result_signal_peaks[-1]]])

    return signal_ac_signal, signal_dc_signal



def Kalman_filter(signal, alpha, beta, noise):
    # Kalman filter to estimate state of system
    kf = simdkalman.KalmanFilter(
        state_transition=np.array([[1, 1], [0, 1]]),
        process_noise=np.diag([alpha, beta]),
        observation_model=np.array([[1, 0]]),
        observation_noise=noise,
    )
    signal = np.array(signal)
    
    # smooth and explain existing data
    smoothed = kf.smooth(signal)
    kalman_1D_signal = smoothed.observations.mean
    return kalman_1D_signal


def find_valid_tidal_volumes(signal):
    # Identify peaks in the signal
    peaks, _ = find_peaks(
        signal,
        width=8,
        distance=18,
        height=np.average(signal),
    )
    # Calculate tidal volumes between successive peaks
    breath_start_indices = peaks[:-1]
    breath_end_indices = peaks[1:]
    breath_signals = [signal[start:end] for start, end in zip(breath_start_indices, breath_end_indices)]
    tidal_volume = [sig[0] - np.min(sig) for sig in breath_signals]

    # Remove outliers in tidal volume data
    tidal_volume = np.array(tidal_volume)
    median_tidal_volume = np.median(tidal_volume)
    threshold = max(7.5, median_tidal_volume * 1.5)

    preprocessed_tidal_volumes = tidal_volume[(tidal_volume >= 2) & (tidal_volume < threshold)]

    # Only include tidal volumes within ±2 std of the mean after outlier removal
    mean_tidal_volume = np.median(preprocessed_tidal_volumes)
    std_tidal_volume = np.std(preprocessed_tidal_volumes)

    valid_indices = np.where(
        (tidal_volume >= 2) & 
        (tidal_volume <= threshold) & 
        (tidal_volume <= mean_tidal_volume + 2 * std_tidal_volume) & 
        (tidal_volume >= mean_tidal_volume - 2 * std_tidal_volume)
    )[0]

    valid_tidal_volumes = tidal_volume[valid_indices]
    valid_peaks = peaks[valid_indices]

    return valid_peaks, valid_tidal_volumes, mean_tidal_volume, std_tidal_volume



def mean_square_diff(matched_ground_truth_signal, matched_reference_signal):
    # Calculate the squared differences directly using vectorized operations
    square_diff = (matched_ground_truth_signal - matched_reference_signal) ** 2
    
    # Compute the mean of the squared differences
    mean_square_diff = np.mean(square_diff)
    
    print(f"MSE: {mean_square_diff:.4f}")

def mean_absolute_diff(matched_ground_truth_signal, matched_reference_signal):
    # Calculate the absolute differences directly using vectorized operations
    absolute_diff = np.abs(matched_ground_truth_signal - matched_reference_signal)
    
    # Compute the mean of the absolute differences
    mean_diff = np.mean(absolute_diff)
    
    # Print the Mean Absolute Difference (MAD)
    print(f"Mean Absolute Difference (MAD): {mean_diff:.4f}")

    # Calculate MAD as a percentage of the mean of the ground truth signal
    mad_percentage = mean_diff / np.mean(matched_ground_truth_signal) * 100

    # Print the MAD as a percentage
    print(f"MAD as Percentage of Ground Truth Mean: {mad_percentage:.2f}%")



def coverage_probability(matched_ground_truth_signal, matched_reference_signal, kappa):
    
    # Calculate the upper and lower bounds
    truth_mean = np.mean(matched_ground_truth_signal)
    margin = kappa / 100 * truth_mean
    truth_upper = matched_ground_truth_signal + margin
    truth_lower = matched_ground_truth_signal - margin
    
    # Count the number of reference signals within the bounds
    count_in = np.sum((matched_reference_signal > truth_lower) & 
                      (matched_reference_signal < truth_upper))
    
    # Calculate the coverage probability
    cp = count_in / len(matched_reference_signal)
    
    # Print the coverage probability
    print(f"CP {kappa}%:", cp)


    
def set_plot_params():
    #Set font parameters and return new colour cycle to use for plotting
    plt.style.use(["default"])
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "text.usetex": False,
        "font.family": "serif",
        "font.sans-serif": "Helvetica",
    }
    colors = sns.color_palette("deep")
    plt.rcParams.update(params)
    return colors

def plot_prettifier(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)