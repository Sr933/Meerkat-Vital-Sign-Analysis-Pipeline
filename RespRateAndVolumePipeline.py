import MeerkatPipelineHelperfunctions
import StatisticalAnalysis
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import sosfiltfilt
from scipy.signal.windows import hamming
from scipy.fft import fft, fftfreq
from scipy.stats import norm
import seaborn as sns


class Respiratory_rate_and_volume_pipeline:
    def __init__(
        self,
        intervall_length_resp=1800,
        intervall_length_vol=1800,
        data_analysis_folder="",
    ):
        """
        Initialize the CalculateHeartRate class with default parameters.

        Args:
            intervall_length (int): Length of the interval in seconds.
            data_analysis_folder (str): Path to the folder containing data to analyze.
        """
        # Path to the data analysis folder
        self.data_analysis_folder = data_analysis_folder

        # Length of the vital sign averaging intervalls
        self.intervall_length_resp = intervall_length_resp
        self.intervall_length_vol = intervall_length_vol

        # Flag to indicate ventilator data presence
        self.has_ventilator = False
        
         # Define the Kalman filter parameters
        self.process_noise = 0.0001
        self.measurement_noise = 0.00001
        self.initial_estimate_error_rate = 10
        self.initial_estimate_error_vol = 4

    def run(self):
        # Function to tie all functions in class together
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(
            self.data_analysis_folder
        )
        self.load__and_process_data()
        self.plot_data()
        # Statistical comparison to ground truth from ventilator only available for some subjects
        if self.has_ventilator:
            self.statistical_analysis()

    def load__and_process_data(self):
        # Load data from camera and ventilator
        self.import_camera_resp_data()
        self.import_ventilator_rate_and_volume()

        # Calulate PCA of depth change signal for frequency estimation
        self.PCA_signal_resp = self.PCA_respiratory_signal_resp()

        # Calculate volume signal for each quadrant from depth changes and ROI size
        self.preprocess_data()

        # Calculate timestamps
        self.generate_timestamps()

        # Bin the measured peaks and volumes into intervals
        self.find_volume_and_rate_in_sliding_window()

        # Calculate resp rate using Fourier analysis of the signal
        self.respiratory_rate_fourier()

        # Estimate the true state of the system with Kalman filters of the measurements
        self.kalman_breathing_rate = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.fourier_breathing_rate, self.process_noise, self.measurement_noise, self.initial_estimate_error_rate
        )
        self.kalman_breathing_rate_peaks = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.intervall_resp_rate, self.process_noise, self.measurement_noise, self.initial_estimate_error_rate
        )

        self.intervall_volumes = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.intervall_volumes, self.process_noise, self.measurement_noise, self.initial_estimate_error_vol
        )

        if self.has_ventilator:
            # Process the obtained ventulator data to make it more comparable to camera signal
            # Process tidal volume best estimates
            self.ventilator_volume_in_window()

            # Process upper and lower bounds
            (
                self.time_in_w,
                self.v_in_w,
                self.v_ex_w,
            ) = self.ventilator_volume_in_window_ex_in(
                self.timestamps_ventilator_volume_upper_lower,
                self.ventilator_volume_inhalation,
                self.ventilator_volume_exhalation,
            )

            # Bin ventilator values to make them comparable
            self.ventilator_rate_windowed_values = (
                MeerkatPipelineHelperfunctions.Kalman_filter(
                    self.ventilator_rate_windowed_values, self.process_noise, self.measurement_noise, self.initial_estimate_error_rate
                )
            )

            # Use the same Kalman filter as on the camera data for comparative results
            self.ventilator_volume_windowed_values = (
                MeerkatPipelineHelperfunctions.Kalman_filter(
                    self.ventilator_volume_windowed_values, self.process_noise, self.measurement_noise, self.initial_estimate_error_vol
                )
            )
            self.v_in_w = MeerkatPipelineHelperfunctions.Kalman_filter(
                self.v_in_w, self.process_noise, self.measurement_noise, self.initial_estimate_error_vol
            )
            self.v_ex_w = MeerkatPipelineHelperfunctions.Kalman_filter(
                self.v_ex_w, self.process_noise, self.measurement_noise, self.initial_estimate_error_vol
            )

    def generate_timestamps(self):
        self.timestamps_intervalls_rate = np.linspace(
            self.ts1[self.intervall_length_resp],
            self.ts1[-1],
            num=int((len((self.PCA_signal_vol)) - self.intervall_length_resp) / 30),
        )
        self.timestamps_intervalls_vol = np.linspace(
            self.ts1[self.intervall_length_vol],
            self.ts1[-1],
            num=int((len((self.PCA_signal_vol)) - self.intervall_length_vol) / 30),
        )

    def preprocess_data(self):
        """
        Preprocesses the respiratory data by performing PCA on depth signals and calculating the flow.

        This function processes depth signals from different regions, applies PCA to obtain a combined
        signal, and identifies valid tidal volumes.
        """
        # Dictionary to store depth signals from different regions
        signals = {
            "left_chest": self.left_chest_depth,
            "right_chest": self.right_chest_depth,
            "left_abdomen": self.left_abdomen_depth,
            "right_abdomen": self.right_abdomen_depth,
        }

        # Apply PCA to each depth signal to obtain respiratory signals
        PCA_signals = {
            key: MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
                depth,
                self.ROI_x_1_array,
                self.ROI_x_2_array,
                self.ROI_y_1_array,
                self.ROI_y_2_array,
            )
            for key, depth in signals.items()
        }

        # Initialize arrays for summing signals and counting valid signal points
        signal_sum = np.zeros_like(PCA_signals["left_abdomen"])
        signal_number = np.zeros_like(PCA_signals["left_abdomen"])

        # Sum signals and count valid signal points for each PCA signal
        for key in PCA_signals:
            signal_sum += np.where(PCA_signals[key] != 10000, PCA_signals[key], 0)
            signal_number += (PCA_signals[key] != 10000).astype(int)

        # Avoid division by zero by setting zero counts to one
        signal_number = np.where(signal_number == 0, 1, signal_number)

        # Calculate the final PCA signal by averaging the valid signals
        self.PCA_signal_vol = signal_sum / signal_number


        # Identify valid tidal volumes and related statistics
        self.valid_peaks, self.valid_tidal_volumes, self.mean, self.std = (
            MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
                self.PCA_signal_vol
            )
        )

    def import_camera_resp_data(self):
        # Load respiratory data from CSV file
        camera_folder = os.path.join(self.subject_folder, "RGB-D camera video data")
        camera_file = os.listdir(camera_folder)[0]
        camera_filepath = os.path.join(camera_folder, camera_file)
        df = pd.read_csv(camera_filepath)

        # Load required data from df into numpy arrays
        self.ts1 = df["Time (s)"].to_numpy()
        self.average_depth = df[" Depth"].to_numpy()
        self.ROI_x_1_array = df[" Rectangle x1"].to_numpy()
        self.ROI_x_2_array = df[" Rectangle x2"].to_numpy()
        self.ROI_y_1_array = df[" Rectangle y1"].to_numpy()
        self.ROI_y_2_array = df[" Rectangle y2"].to_numpy()
        self.left_chest_depth = df[" Depth Left Chest"].to_numpy()
        self.right_chest_depth = df[" Depth Right Chest"].to_numpy()
        self.left_abdomen_depth = df[" Depth Left Abdomen"].to_numpy()
        self.right_abdomen_depth = df[" Depth Right Abdomen"].to_numpy()

    def import_ventilator_rate_and_volume(self):
        # Import rate
        rate_folder = os.path.join(self.subject_folder, "Ventilator respiratory rate")

        # Read ventilator data if available in database
        if os.listdir(rate_folder):
            self.has_ventilator = True

            # Import ventilator rate from csv
            rate_file = os.listdir(rate_folder)[0]
            rate_filepath = os.path.join(rate_folder, rate_file)
            df = pd.read_csv(rate_filepath)

            self.timestamps_ventilator_rate = df["Time (s)"].to_numpy()
            self.ventilator_rate = df[" Respiratory rate (breaths/min)"].to_numpy()

            # Import tidal volume best estimates from ventilator from csv
            volume_folder = os.path.join(
                self.subject_folder, "Ventilator tidal volume best estimate"
            )
            volume_file = os.listdir(volume_folder)[0]
            volume_filepath = os.path.join(volume_folder, volume_file)
            df2 = pd.read_csv(volume_filepath)

            self.timestamps_ventilator_volume_best_estimate = df2["Time (s)"].to_numpy()
            self.ventilator_tidal_volume_best_estimate = df2[" VT (ml)"].to_numpy()

            # Import upper lower estimation for tidal volume from measurements of inhalation and exhalation from csv
            volume_upper_lower_folder = os.path.join(
                self.subject_folder, "Ventilator tidal volume upper lower"
            )
            volume_upper_lower_file = os.listdir(volume_upper_lower_folder)[0]
            volume_upper_lower_filepath = os.path.join(
                volume_upper_lower_folder, volume_upper_lower_file
            )
            df3 = pd.read_csv(volume_upper_lower_filepath)

            self.timestamps_ventilator_volume_upper_lower = df3["Time (s)"].to_numpy()
            self.ventilator_volume_exhalation = df3[" VT_Expiration (ml)"].to_numpy()
            self.ventilator_volume_inhalation = df3[" VT_Inspiration (ml)"].to_numpy()

    def PCA_respiratory_signal_resp(self):
        # Bandpass filter signal followed by timeseries PCA using Hankel matrix
        butter = MeerkatPipelineHelperfunctions.butter_bandpass(
            15 / 60, 150 / 60, 30, order=7
        )
        filtered_data = sosfiltfilt(butter, self.average_depth)
        F_trend = MeerkatPipelineHelperfunctions.PCA_timeseries_resp_signal(
            filtered_data
        )
        return F_trend

    def find_volume_and_rate_in_sliding_window(self):
        self.intervall_volumes = []
        self.intervall_resp_rate = []

        # Calculate the number of intervals
        num_intervals = int((len(self.PCA_signal_vol) - self.intervall_length_vol) / 30)

        # Initialize arrays for respiratory rates and volumes
        self.intervall_resp_rate = np.zeros(num_intervals)
        self.intervall_volumes = np.zeros(num_intervals)

        # Iterate through each interval
        for i in range(num_intervals):
            # Determine the range of peaks within the current interval
            start_idx = 30 * i
            end_idx = self.intervall_length_vol + 30 * i
            result = np.where((self.valid_peaks >= start_idx) & (self.valid_peaks <= end_idx))[0]

            # Calculate respiratory rate and volume if peaks are found
            if len(result) > 1:
                first_peak = self.valid_peaks[result[0]]
                last_peak = self.valid_peaks[result[-1]]
                self.intervall_resp_rate[i] = (len(result) - 1) / (last_peak - first_peak) * 1800 #60s * 30 Hz
                self.intervall_volumes[i] = np.mean(self.valid_tidal_volumes[result])
            else:
                # Assign previous values or defaults if no peaks are found
                self.intervall_resp_rate[i] = self.intervall_resp_rate[i-1] if i > 0 else 0
                self.intervall_volumes[i] = self.intervall_volumes[i-1] if i > 0 else 0

    def respiratory_rate_fourier(self):
        self.fourier_breathing_rate = []
        peaks_in_bins = []
        
        num_intervalls = int(
            ((len(self.PCA_signal_resp)) - self.intervall_length_resp) / 30
        )
        for i in range(num_intervalls):

            # Calculate Hamming window signal
            window_signal = self.PCA_signal_resp[
                30 * i : 30 * i + self.intervall_length_resp
            ]
            n = len(window_signal)
            w = hamming(n)

            # Perform fourier analysis
            t = np.arange(n)
            yf = np.abs(fft(window_signal * w))
            xf = fftfreq(t.shape[-1], 1 / 1800)

            ##Perform inference on resp rate using prior of expected frequency and
            # likelihood as fourier frequency
            peaks_in_bins.append(xf[np.where(yf == np.max(yf))[0][0]])
            if (
                len(self.fourier_breathing_rate) > 10
            ):  # Calculate adaptive prior from know distribution and previous distribution of the individual baby
                m = np.mean(peaks_in_bins)
                s = np.std(peaks_in_bins) + 0.1  # for numerical stability

                mean = 50 + (1) / (1 + (s / 30)) * (m - 50)
            else:  # Prior based on known distribution of breaths
                mean = 50
                std = 20
                
                
            resp_gaussian = norm.pdf(xf, mean, std)
            yf = np.multiply(yf, resp_gaussian)
            self.fourier_breathing_rate.append(xf[np.where(yf == np.max(yf))[0][0]])

    def ventilator_volume_in_window(self):
        # Evaluate ventilator volumes in sliding window of same length as the volumes obtained from the camera

        vent_time_start = self.timestamps_ventilator_volume_best_estimate[0]
        vent_time_end = self.timestamps_ventilator_volume_best_estimate[-1]
        self.ventilator_volume_windowed_values = []
        self.timestamps_ventilator_volume_windowed_values = []
        self.ventilator_rate_windowed_values = []

        # Iterate over signal to bin values and average them
        for i in range(int(vent_time_end - vent_time_start)):
            # Find values in window
            window_start_v = vent_time_start + i
            window_end_v = vent_time_start + i + int(self.intervall_length_vol / 30)
            result_v = np.where(
                np.logical_and(
                    self.timestamps_ventilator_volume_best_estimate >= window_start_v,
                    self.timestamps_ventilator_volume_best_estimate <= window_end_v,
                )
            )[0]
            # Add timestamp of window and mean of values in it for tidal volumes
            self.timestamps_ventilator_volume_windowed_values.append(window_end_v)
            self.ventilator_volume_windowed_values.append(
                np.mean(
                    self.ventilator_tidal_volume_best_estimate[
                        result_v[0] : result_v[-1]
                    ]
                )
            )
            # Add timestamp of window and mean of values in it for respiratory rates
            window_start_r = vent_time_start + i
            window_end_r = vent_time_start + i + int(self.intervall_length_resp / 30)

            result_r = np.where(
                np.logical_and(
                    self.timestamps_ventilator_rate >= window_start_r,
                    self.timestamps_ventilator_rate <= window_end_r,
                )
            )[0]
            self.ventilator_rate_windowed_values.append(
                np.mean(self.ventilator_rate[result_r[0] : result_r[-1]])
            )

    def ventilator_volume_in_window_ex_in(self, timestamps, volumes_in, volumes_ex):
        # Evaluate ventilator volumes in sliding window of same length as the volumes obtained from the camera

        vent_time_start = timestamps[0]
        vent_time_end = timestamps[-1]
        window_timestamps = []
        window_volumes_in = []
        window_volumes_ex = []

        # Iterate over signal to bin values and average them
        for i in range(int(vent_time_end - vent_time_start)):
            window_start = vent_time_start + i
            window_end = vent_time_start + i + int(self.intervall_length_vol / 30)
            result_v = np.where(
                np.logical_and(
                    timestamps >= window_start,
                    timestamps <= window_end,
                )
            )[0]
            # Tidal volumes found in window add the timestamp of the window and the mean of the volumes
            if len(result_v) > 0:
                window_timestamps.append(window_end)
                window_volumes_in.append(
                    np.mean(volumes_in[result_v[0] : result_v[-1]])
                )
                window_volumes_ex.append(
                    np.mean(volumes_ex[result_v[0] : result_v[-1]])
                )

        return window_timestamps, window_volumes_in, window_volumes_ex

    def plot_data(self):
        # Plot parameters
        colors = MeerkatPipelineHelperfunctions.set_plot_params()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot ventilator tidal volume best estimates and upper lower bounds if available
        if self.has_ventilator:
            ax1.plot(
                np.array(self.timestamps_ventilator_volume_windowed_values) / 60,
                self.ventilator_volume_windowed_values,
                label="Ventilator",
                color=colors[0],
            )

            ax1.fill_between(
                np.array(self.time_in_w) / 60,
                self.v_in_w,
                self.v_ex_w,
                color=colors[0],
                alpha=0.4,
                label="Ventilator bounds",
            )
        # Plot camera estimates
        ax1.plot(
            np.array(self.timestamps_intervalls_vol) / 60,
            self.intervall_volumes,
            label="Camera",
            color=colors[1],
        )

        ax1.set_xlabel("Time (min)", fontsize=20)
        ax1.set_ylabel("Tidal volume (ml)", fontsize=20)
        ax1.set_ylim(2, 9)
        ax1.set_title("Tidal volume", fontsize=20)
        self.resp_plot_params(ax1)

        # Plot ventilator rate if available
        if self.has_ventilator:
            ax2.plot(
                np.array(self.timestamps_ventilator_volume_windowed_values) / 60,
                self.ventilator_rate_windowed_values,
                color=colors[0],
                label="Ventilator",
            )
        # Plot camera respiratory rate
        ax2.plot(
            np.array(self.timestamps_intervalls_rate) / 60,
            self.kalman_breathing_rate,
            label="Camera",
            color=colors[1],
        )
        ax2.set_xlabel("Time (min)", fontsize=20)
        ax2.set_ylabel("Respiratory rate (breaths/min)", fontsize=20)
        ax2.set_title("Respiratory rate", fontsize=20)
        ax2.legend(loc="upper left", fontsize=14)
        ax2.set_ylim(15, 80)
        self.resp_plot_params(ax2)

        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def resp_plot_params(self, ax):
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(loc="upper right", frameon=False, fontsize=14)

    def statistical_analysis(self):
        """
        Perform statistical analysis for different measurement and estimation techniques.
        """

        # Define a helper function to run statistical analysis
        def run_analysis(
            vital_sign,
            ground_truth_signal,
            ground_truth_timestamps,
            reference_signal,
            reference_timestamps,
            **kwargs,
        ):
            print(f"{vital_sign.replace('_', ' ').title()}")
            analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
            analysis.vital_sign = vital_sign
            analysis.subject_folder = self.subject_folder
            analysis.ground_truth_signal = ground_truth_signal
            analysis.ground_truth_timestamps = ground_truth_timestamps
            analysis.reference_signal = reference_signal
            analysis.reference_timestamps = reference_timestamps
            for key, value in kwargs.items():
                setattr(analysis, key, value)
            analysis.run()

        # Perform Fourier analysis
        run_analysis(
            vital_sign="Resp_Fourier",
            ground_truth_signal=self.ventilator_rate_windowed_values,
            ground_truth_timestamps=self.timestamps_ventilator_volume_windowed_values,
            reference_signal=self.kalman_breathing_rate,
            reference_timestamps=self.timestamps_intervalls_rate,
        )

        # Perform Resp Peak counting
        run_analysis(
            vital_sign="Resp_Peak_counting",
            ground_truth_signal=self.ventilator_rate_windowed_values,
            ground_truth_timestamps=self.timestamps_ventilator_volume_windowed_values,
            reference_signal=self.kalman_breathing_rate_peaks,
            reference_timestamps=self.timestamps_intervalls_vol,
        )

        # Perform Tidal volume best estimate
        run_analysis(
            vital_sign="Tidal_volume",
            ground_truth_signal=self.ventilator_volume_windowed_values,
            ground_truth_timestamps=self.timestamps_ventilator_volume_windowed_values,
            reference_signal=self.intervall_volumes,
            reference_timestamps=self.timestamps_intervalls_vol,
        )

        # Perform Tidal volume upper lower analysis
        run_analysis(
            vital_sign="Tidal_volume_upper_lower",
            ground_truth_timestamps=self.time_in_w,
            reference_signal=self.intervall_volumes,
            reference_timestamps=self.timestamps_intervalls_vol,
            ground_truth_signal=self.v_ex_w,
            tidalvolumeupper=self.v_ex_w,
            tidalvolumelower=self.v_in_w,
            tidalvolumeflag=True,
        )

    def return_data(self):
        """
        Returns:
            dict: A dictionary with the following key-value pairs:
                - "Timestamps Ventilator Volume Windowed Values" (array): Timestamps for ventilator volume with windowing.
                - "Ventilator Volume Windowed Values" (array): Windowed values of ventilator volume.
                - "Ventilator Rate Windowed Values" (array): Windowed values of ventilator rate.
                - "Time in W" (array): Time data in the window.
                - "V In W" (array): Volume data in the window (in).
                - "V Ex W" (array): Volume data in the window (ex).
                - "Timestamps Intervals Volume" (array): Timestamps for volume intervals.
                - "Interval Volumes" (array): Volume data for intervals.
                - "Timestamps Intervals Rate" (array): Timestamps for rate intervals.
                - "Kalman Breathing Rate" (array): Kalman-filtered breathing rate.
                - "Kalman Breathing Rate Peaks" (array): Peaks in the Kalman-filtered breathing rate.
        """

        # Create a dictionary with all the relevant data
        data = {
            "Timestamps Ventilator Volume Windowed Values": self.timestamps_ventilator_volume_windowed_values,
            "Ventilator Volume Windowed Values": self.ventilator_volume_windowed_values,
            "Ventilator Rate Windowed Values": self.ventilator_rate_windowed_values,
            "Time in W": self.time_in_w,
            "V In W": self.v_in_w,
            "V Ex W": self.v_ex_w,
            "Timestamps Intervals Volume": self.timestamps_intervalls_vol,
            "Interval Volumes": self.intervall_volumes,
            "Timestamps Intervals Rate": self.timestamps_intervalls_rate,
            "Kalman Breathing Rate": self.kalman_breathing_rate,
            "Kalman Breathing Rate Peaks": self.kalman_breathing_rate_peaks,
        }

        return data
