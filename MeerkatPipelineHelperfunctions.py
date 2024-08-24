import os
import numpy as np
from scipy.signal import find_peaks
import simdkalman
import scipy
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    
    
'''
Bland-Altman mean-difference plots

Author: Joses Ho
License: BSD-3
'''

import numpy as np

import utils



def mean_diff_plot(m1, m2, color, sd_limit=1.96, ax=None):
    """
    Construct a Tukey/Bland-Altman Mean Difference Plot.

    Tukey's Mean Difference Plot (also known as a Bland-Altman plot) is a
    graphical method to analyze the differences between two methods of
    measurement. The mean of the measures is plotted against their difference.

    For more information see
    https://en.wikipedia.org/wiki/Bland-Altman_plot

    Parameters
    ----------
    m1 : array_like
        A 1-d array.
    m2 : array_like
        A 1-d array.
    sd_limit : float
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted are md +/- sd_limit * sd.
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences. If sd_limit = 0, no limits will be plotted, and
        the ylimit of the plot defaults to 3 standard deviations on either
        side of the mean.
    ax : AxesSubplot
        If `ax` is None, then a figure is created. If an axis instance is
        given, the mean difference plot is drawn on the axis.
    scatter_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    References
    ----------
    Bland JM, Altman DG (1986). "Statistical methods for assessing agreement
    between two methods of clinical measurement"

    Examples
    --------

    Load relevant libraries.

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Making a mean difference plot.

    >>> # Seed the random number generator.
    >>> # This ensures that the results below are reproducible.
    >>> np.random.seed(9999)
    >>> m1 = np.random.random(20)
    >>> m2 = np.random.random(20)
    >>> f, ax = plt.subplots(1, figsize = (8,5))
    >>> sm.graphics.mean_diff_plot(m1, m2, ax = ax)
    >>> plt.show()

    .. plot:: plots/graphics-mean_diff_plot.py
    """

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError(f'sd_limit ({sd_limit}) is less than 0.')

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    
    spacing=60
    ax.scatter(means[::spacing], diffs[::spacing], color=color, s=2.5) # Plot the means against the diffs.
    ax.axhline(mean_diff, color="k", linewidth=0.5)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate(f'mean diff:\n{np.round(mean_diff, 2)}',
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=8,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, color="k", linewidth=0.5)
        ax.annotate(f'-{sd_limit} SD: {lower:0.2g}',
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=8,
                    xycoords='axes fraction')
        ax.annotate(f'+{sd_limit} SD: {upper:0.2g}',
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=8,
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    


def vital_sign_summary_statistics(data_analysis_folder, vital_sign):
    # Get all subjects in dataset
    
    subjects = []
    truth = []
    camera = []
    subject_number = 0
    for file in os.listdir(data_analysis_folder):
        if "mk" in file:
            subjects.append(os.fsdecode(file))

    # Iterate over subjects to import all data
    for subject in subjects:
        # Define subject folder
        subject_folder=os.path.join(data_analysis_folder, subject)
        signal_folder=os.path.join(subject_folder, "Pipeline results")

        # Try to import file, if not present skip the subject
        try:
            if os.listdir(signal_folder):
                # Iterate over all files available to find the matching intermediate file
                for file in os.listdir(signal_folder):
                    if vital_sign in file:
                        signal_filepath=os.path.join(signal_folder, file)

                        # Iterate over file
                        # Iterate over file
                        df=pd.read_csv(signal_filepath)
                        tr=df["Ground Truth Signal"].to_list()
                        cam=df["Reference Signal"].to_list()
                        
                        [truth.append(t) for t in tr]
                        [camera.append(c) for c in cam]
                        subject_number += 1
                        break

        except:
            continue
    print(vital_sign)
    print("Number of subjects: ", subject_number)
    truth=np.asarray(truth)
    camera=np.asarray(camera)
    ##Summary statistics

    kappa = 10 if "saturation" not in vital_sign else 3
    if "POS" in vital_sign or "CHROM" in vital_sign:
        kappa=5
    coverage_probability(truth, camera, kappa=kappa)
    kappa = kappa * 2
    coverage_probability(truth, camera, kappa=kappa)
    mean_absolute_diff(truth, camera)
    mean_square_diff(truth, camera)

    # Plot data
    colors=set_plot_params()
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    # BA plot
    mean_diff_plot(camera, truth, ax=axs[0, 0], color=colors[0])
    axs[0, 0].set_ylabel("Difference (/min)", fontsize=20)
    axs[0, 0].set_xlabel("Means (/min)", fontsize=20)
    axs[0, 0].set_title("(a) Bland Altman plot", fontsize=20)
    plot_prettifier(axs[0,0])


    ### Difference distribution
    difference = camera - truth
    bins = (
        range(int(min(difference)), int(max(difference)) + 1, 1)
        if "Tidal" not in vital_sign
        else np.linspace(int(min(difference)), int(max(difference)), 36)
    )
    axs[0, 1].hist(difference, bins=bins, color=colors[0])
    axs[0, 1].set_xlabel("Camera - Ground Truth (/min)", fontsize=20)
    axs[0, 1].set_ylabel("Frequency", fontsize=20)
    axs[0, 1].set_title("(b) Histogram of errors", fontsize=20)
    plot_prettifier(axs[0,1])
   


    # Correlation plot with linear regression
    spacing=60
    axs[1, 0].scatter(truth[::spacing], camera[::spacing], color=colors[0],s=2.5)
    axs[1, 0].set_ylabel("Camera (/min)", fontsize=20)
    axs[1, 0].set_xlabel("Ground Truth (/min)", fontsize=20)
    axs[1, 0].set_xlim(
        min(min(truth), min(camera)) - 3, max(max(truth), max(camera)) + 3
    )
    axs[1, 0].set_ylim(
        min(min(truth), min(camera)) - 3, max(max(truth), max(camera)) + 3
    )
    axs[1, 0].plot([0, 200], [00, 200], "k--", label="ideal fit")
    axs[1, 0].legend(loc="upper left", fontsize=14, frameon=False)
    axs[1, 0].set_title("(c) Correlation", fontsize=20)
    plot_prettifier(axs[1,0])

    # Calculate the coverage probability as a function of error
    
    
    k_diff = 0.1 if  "Tidal" in vital_sign else 1
    k=k_diff
    kappas=[]
    coverage_probability_array = []
    i=0
    while True:
            cp = coverage_probability_abs(truth, camera, k) * 100
            coverage_probability_array.append(cp)
            kappas.append(k)
            i += 1
            k += k_diff
            if cp > 95:
                break
           

    axs[1, 1].plot(kappas, coverage_probability_array, color=colors[0])
    axs[1, 1].set_xlabel("Absolute error (/min)", fontsize=20)
    axs[1, 1].set_ylabel("Coverage probability (/min)", fontsize=20)
    axs[1, 1].set_title("(d) Coverage probability", fontsize=20)
    plot_prettifier(axs[1,1])


    fig.align_ylabels()
    plt.tight_layout(pad=2.5)
    plt.show()


def coverage_probability_abs(truth, test, kappa):
    # Calculate proportion of values lying between the upper and lower acceptbale bounds
    truth_upper = truth + kappa
    truth_lower = truth - kappa
    # Assuming truth, truth_lower, truth_upper, and test are numpy arrays
    mask = (test > truth_lower) & (test < truth_upper)
    count_in = np.sum(mask)
    tdi = count_in / len(truth)
    return tdi




