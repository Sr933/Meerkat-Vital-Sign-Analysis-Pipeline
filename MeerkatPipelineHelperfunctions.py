import os
import numpy as np
from scipy.signal import find_peaks
import simdkalman
import scipy
from sklearn import decomposition

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
    F_trend = np.multiply(F_trend, chest_area) / 10**3

    # Assign 10000 to outlier values to remove them in later processing

    for i in range(len(F_trend)):
        if abs(F_trend[i]) > 8:
            F_trend[i] = 10000 if outliers else 8
    return F_trend


def choose_subject(data_analysis_folder):

    filenames = []
    i = 1
    for file in os.listdir(data_analysis_folder):
        filenames.append(os.fsdecode(file))
        if "mk" in file:
            print(i, " : ", file)
            i += 1

    input_file = False
    while not input_file:
        users_choice = input("Desired subject: ")
        if users_choice.isnumeric():
            users_choice = int(users_choice) - 1
            if users_choice <= len(filenames):
                file_folder = filenames[users_choice]
                input_file = True
                continue

            else:
                print("Out of range")

        else:
            print("Input must be a number")

    file_folder = repr("\\" + file_folder)
    file_folder = file_folder[2:-1]
    subject_folder = data_analysis_folder + file_folder
    return subject_folder


def find_ac_signal(filtered_signal, intervall):
    signal_ac_signal = []
    signal_dc_signal = []
    filtered_signal = np.array(filtered_signal)
    peaks_signal, _ = find_peaks(filtered_signal, distance=6)

    valleys_signal, _ = find_peaks(-filtered_signal, distance=6)

    # Calculate peaks and valleys in each intervall length
    for time in range(int((len(filtered_signal) - intervall) / 30)):
        result_signal_peaks = np.where(
            np.logical_and(
                peaks_signal >= 30 * time, peaks_signal <= intervall + 30 * time
            )
        )[0]
        result_signal_valleys = np.where(
            np.logical_and(
                valleys_signal >= 30 * time, valleys_signal <= intervall + 30 * time
            )
        )[0]
        peaks = []
        valleys = []
        for peak in range(len(result_signal_peaks)):
            peaks.append(filtered_signal[peaks_signal[peak]])

        for valley in range(len(result_signal_valleys)):
            valleys.append(filtered_signal[valleys_signal[valley]])
        amplitudes = []

        # Calculate amplitude between peaks and valleys
        if len(peaks) > len(valleys):
            for i in range(len(valleys)):
                amplitudes.append(peaks[i] - valleys[i])
        elif len(peaks) < len(valleys):
            for i in range(len(peaks)):
                amplitudes.append(peaks[i] - valleys[i])

        else:
            amplitudes = np.array(peaks) - np.array(valleys)

        signal_ac_signal.append(np.median(amplitudes))
        signal_dc_signal.append(
            np.median(
                filtered_signal[
                    peaks_signal[result_signal_peaks[0]] : peaks_signal[
                        result_signal_peaks[-1]
                    ]
                ]
            )
        )
    return signal_ac_signal, signal_dc_signal


def Kalman_filter(signal, alpha, beta, noise):
    # Kalman filter to estimate state of system
    kf = simdkalman.KalmanFilter(
        state_transition=np.array([[1, 1], [0, 1]]),
        process_noise=np.diag([alpha, beta]),
        observation_model=np.array([[1, 0]]),
        observation_noise=noise,
    )

    data = np.array(signal)
    # smooth and explain existing data
    smoothed = kf.smooth(data)
    kalman_1D_signal = smoothed.observations.mean
    return kalman_1D_signal


def find_valid_tidal_volumes(timestamps, signal):
    # Asess whether motion is likely to be a breath or not by comparing tidal volume with all tidal volumes in recording
    breathing_rate = []
    breath_timestamps = []
    peak_location = []

    peaks, _ = find_peaks(
        signal,
        width=8,
        distance=18,
        height=np.average(np.array(signal)),
    )
    for i in range(len(peaks) - 1):
        breathing_rate.append(1800 / (peaks[i + 1] - peaks[i]))
        breath_timestamps.append(timestamps[peaks[i]])
        peak_location.append(timestamps[peaks[i]])

    tidal_volume = []

    for i in range(len(peaks) - 1):
        breath_start = peaks[i]
        breath_end = peaks[i + 1]
        breath_signal = signal[breath_start:breath_end]
        maximum_depth = breath_signal[0]
        minimum_depth = np.min(breath_signal)
        tidal = maximum_depth - minimum_depth
        tidal_volume.append(tidal)

    # remove outliers in data
    preprocessed_tidal_volumes = []
    med = np.median(tidal_volume)
    threshold = 7.5
    if med > 5:
        threshold = med * 1.5
    for vol in tidal_volume:
        if vol >= 2 and vol < threshold:
            preprocessed_tidal_volumes.append(vol)

    # only include +- 2 std of mean volume after outlier removal
    mean = np.median(preprocessed_tidal_volumes)
    std = np.std(preprocessed_tidal_volumes)

    valid_tidal_volumes = []
    valid_peaks = []

    for i in range(len(tidal_volume)):
        if (
            tidal_volume[i] >= 2
            and tidal_volume[i] <= threshold
            and tidal_volume[i] <= mean + 2 * std
            and tidal_volume[i] >= mean - 2 * std
        ):
            valid_tidal_volumes.append(tidal_volume[i])
            valid_peaks.append(peaks[i])
    return np.asarray(valid_peaks), np.asarray(valid_tidal_volumes), mean, std


def mean_square_diff(matched_ground_truth_signal, matched_reference_signal):
    square_diff = np.zeros(len(matched_ground_truth_signal))
    for i in range(len(matched_ground_truth_signal)):
        square_diff[i] = (
            matched_ground_truth_signal[i] - matched_reference_signal[i]
        ) ** 2
    mean_square_diff = np.mean(square_diff)
    print("MSE:", mean_square_diff)


def mean_absolute_diff(matched_ground_truth_signal, matched_reference_signal):
    absolute_diff = np.zeros(len(matched_ground_truth_signal))
    for i in range(len(matched_ground_truth_signal)):
        absolute_diff[i] = abs(
            matched_ground_truth_signal[i] - matched_reference_signal[i]
        )
    mean_diff = np.mean(absolute_diff)
    print("MAD:", mean_diff)
    print("MAD:", mean_diff/np.mean(matched_ground_truth_signal)*100, "%")


def coverage_probability(matched_ground_truth_signal, matched_reference_signal, kappa):
    truth = np.array(matched_ground_truth_signal)
    truth_upper = truth + kappa / 100 * np.mean(truth)
    truth_lower = truth - kappa / 100 * np.mean(truth)
    count_in = 0
    for i in range(len(matched_reference_signal)):
        if (
            matched_reference_signal[i] > truth_lower[i]
            and matched_reference_signal[i] < truth_upper[i]
        ):
            count_in += 1

    cp = count_in / len(matched_reference_signal)
    print("CP", kappa, "%:", cp)
