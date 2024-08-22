import numpy as np
import matplotlib.pyplot as plt
import MeerkatPipelineHelperfunctions
import os
from scipy.signal import sosfiltfilt
import pandas as pd
from sklearn import decomposition
import StatisticalAnalysis
import sys

class CalculatePulseOxygenation:
    def __init__(self, data_analysis_folder="", interval_length=900):
        """
        Initializes the CalculatePulseOxygenation class with the given parameters.

        Args:
            data_analysis_folder (str): Path to the folder containing data for analysis.
            interval_length (int): Length of the interval for processing.
        """
        self.data_analysis_folder = data_analysis_folder
        self.interval_length = interval_length

        # Define the Kalman filter parameters
        self.process_noise = 0.0001
        self.measurement_noise = 0.00001
        self.initial_estimate_error = 10

    def run(self):
        # Function to tie all functions in class together
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(
            self.data_analysis_folder
        )
        self.import_oxygen_saturation_data()
        self.smooth_signals()
        self.import_oximeter_spO2()
        self.infrared_oxygenation()
        self.rgb_oxygenation()
        self.ycgcr_oxygenation()
        self.PCA_oxygenation()
        self.generate_timestamps()
        self.plot_data()
        self.statistical_analysis()

    def import_oxygen_saturation_data(self):
        # Define camera filepath
        camera_folder = os.path.join(self.subject_folder, "RGB-D camera video data")
        camera_file = os.path.join(camera_folder, os.listdir(camera_folder)[0])

        # Load data from csv file
        df = pd.read_csv(camera_file)
        self.ts1 = df["Time (s)"].to_numpy()
        self.red_signal = df[" Red"].to_numpy()
        self.green_signal = df[" Green"].to_numpy()
        self.blue_signal = df[" Blue"].to_numpy()
        self.infrared_signal = df[" IR"].to_numpy()

    def smooth_signals(self):
        # Calculate average signals
        self.mean_red = np.mean(self.red_signal)
        self.mean_blue = np.mean(self.blue_signal)
        self.mean_green = np.mean(self.green_signal)
        self.mean_infrared = np.mean(self.infrared_signal)

        # Bandpass filter the signals
        self.ac_filter = MeerkatPipelineHelperfunctions.butter_bandpass(
            1, 5, 30, order=7
        )
        self.filtered_red = sosfiltfilt(self.ac_filter, self.red_signal) + self.mean_red
        self.filtered_blue = (
            sosfiltfilt(self.ac_filter, self.blue_signal) + self.mean_blue
        )
        self.filtered_green = (
            sosfiltfilt(self.ac_filter, self.green_signal) + self.mean_green
        )
        self.filtered_ir = (
            sosfiltfilt(self.ac_filter, self.infrared_signal) + self.mean_infrared
        )

    def import_oximeter_spO2(self):
        # Define pulse oximeter folder
        oximeter_folder = os.path.join(
            self.subject_folder, "Pulse oximeter oxygen saturation"
        )
        if os.listdir(oximeter_folder):  # if file exists import data from it
            oximeter_file = os.path.join(
                oximeter_folder, os.listdir(oximeter_folder)[0]
            )

            df = pd.read_csv(oximeter_file)
            self.pulse_oximeter_timestamps = df["Time (s)"].to_numpy()
            self.pulse_oximeter_spO2 = df[" Oxygen saturation (%)"].to_numpy()
            self.pulse_oximeter_spO2 = MeerkatPipelineHelperfunctions.Kalman_filter(
                self.pulse_oximeter_spO2,
                self.process_noise,
                self.measurement_noise,
                self.initial_estimate_error,
            )
        else:
            print("No valid pulse oximeter data")
            sys.exit()

    def rgb_oxygenation(self):
        # Algorithm after https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4574660/

        # Get AC and DC components
        red_ac_signal, red_dc_signal = MeerkatPipelineHelperfunctions.find_ac_signal(
            self.filtered_red, self.intervall_length
        )
        blue_ac_signal, blue_dc_signal = MeerkatPipelineHelperfunctions.find_ac_signal(
            self.filtered_blue, self.intervall_length
        )
        # Calculate ratio of ratios
        red_signal = np.divide(red_ac_signal, red_dc_signal)
        blue_signal = np.divide(blue_ac_signal, blue_dc_signal)
        self.rr_signal_rgb = np.divide(blue_signal, red_signal)

        self.spO2_rgb = (
            40 * np.log(self.rr_signal_rgb) + 80
        )  # calibration can be adjusted as needed

        self.spO2_rgb = self.clamp_spO2(self.spO2_rgb)
        # Smooth using Kalman filter
        self.spO2_rgb = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_rgb,
            self.process_noise,
            self.measurement_noise,
            self.initial_estimate_error,
        )

    def ycgcr_oxygenation(self):
        # Algorithm after https://www.mdpi.com/1424-8220/21/18/6120

        # Normalize data to 0:1 range
        r_normalized = self.red_signal / 255
        g_normalized = self.green_signal / 255
        b_normalized = self.blue_signal / 255
        # Calculate YCgCr colour space, not Y component not needed for further analysis
        # Y=16+ (65.481 * r_normalized)+(128.533 * g_normalized)+(24.966 * b_normalized)
        Cg = (
            128
            + (-81.085 * r_normalized)
            + (112 * g_normalized)
            + (-30.915 * b_normalized)
        )
        Cr = (
            128
            + (112 * r_normalized)
            + (-93.786 * g_normalized)
            + (-18.214 * b_normalized)
        )

        # Bandpass filter signals
        filtered_cg = sosfiltfilt(self.ac_filter, Cg) + np.mean(Cg)
        filtered_cr = sosfiltfilt(self.ac_filter, Cr) + np.mean(Cr)

        # Calculate AC and DC components
        cg_ac_signal, _ = MeerkatPipelineHelperfunctions.find_ac_signal(
            filtered_cg, self.intervall_length
        )
        cr_ac_signal, _ = MeerkatPipelineHelperfunctions.find_ac_signal(
            filtered_cr, self.intervall_length
        )
        # Excliude outliers in measurements
        self.rr_ycgcr = np.divide(np.log(cr_ac_signal), np.log(cg_ac_signal))
        # Create a boolean mask for out-of-range values
        out_of_range = (self.rr_ycgcr < 0) | (self.rr_ycgcr > 1.8)
        
        # Replace out-of-range values with NaN for interpolation
        self.rr_ycgcr[out_of_range] = np.nan

        # Forward fill the NaN values
        self.rr_ycgcr = pd.Series(self.rr_ycgcr).ffill().to_numpy()

        # Replace any remaining NaN (which would be the very first element if it was out-of-range)
        self.rr_ycgcr[np.isnan(self.rr_ycgcr)] = 1
        # If values over 100 or below 75 truncate
        self.spO2_ycgcr = 11.88 * self.rr_ycgcr + 82  # calibration adjustable

        self.spO2_ycgcr = self.clamp_spO2(self.spO2_ycgcr)
        # Smooth using Kalman filter
        self.spO2_ycgcr = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_ycgcr,
            self.process_noise,
            self.measurement_noise,
            self.initial_estimate_error,
        )

    def infrared_oxygenation(self):
        # Algorithm after https://ieeexplore.ieee.org/ielaam/10/7471384/7275158-aam.pdf?tag=1

        # Get AC and DC components in signal
        red_ac_signal, red_dc_signal = MeerkatPipelineHelperfunctions.find_ac_signal(
            self.filtered_red, self.intervall_length
        )
        ir_ac_signal, ir_dc_signal = MeerkatPipelineHelperfunctions.find_ac_signal(
            self.filtered_ir, self.intervall_length
        )

        # Calculate ratio of ratios
        red_signal = np.divide(red_ac_signal, red_dc_signal)
        ir_signal = np.divide(ir_ac_signal, ir_dc_signal)
        self.rr_signal_infrared = np.divide(red_signal, ir_signal)

        # Calculate SpO2 truncate at 100
        self.spO2_infrared = 100 + 20 / 6 - 20 / 3 * self.rr_signal_infrared
        
        self.spO2_infrared=self.clamp_spO2(self.spO2_infrared)
        self.spO2_infrared = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_infrared,
            self.process_noise,
            self.measurement_noise,
            self.initial_estimate_error,
        )
    def generate_timestamps(self):
        # Define timestamps for bins
        self.time_spO2 = np.linspace(
            self.ts1[self.intervall_length],
            self.ts1[-1],
            num=int((len((self.filtered_red)) - self.intervall_length) / 30),
        )
        self.timestamps_spO2_calibrated = np.linspace(
            self.ts1[self.intervall_length], self.ts1[-1], num=len(self.spO2_calibrated)
        )
    
    def PCA_oxygenation(self):
        # Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10712673/
        # Perform signal extraction using PCA from shallow layer
        shallow_signal = np.stack(
            (self.filtered_red, self.filtered_ir, self.filtered_green)
        )
        estimator = decomposition.PCA(n_components=2, svd_solver="full")
        estimator.fit(shallow_signal)
        shallow_noise = np.array(estimator.components_[0])

        # Perform signal extraction using PCA from deep layer
        infrared_with_shallow = np.stack((self.filtered_ir, shallow_noise))
        red_with_shallow = np.stack((self.filtered_red, shallow_noise))

        estimator_red = decomposition.PCA(n_components=2, svd_solver="full")
        estimator_red.fit(red_with_shallow)
        red_signal = estimator_red.components_[1][100:]

        estimator_infrared = decomposition.PCA(n_components=2, svd_solver="full")
        estimator_infrared.fit(infrared_with_shallow)
        infrared_signal = estimator_infrared.components_[1][100:]

        # Calculate AC and DC final and ratio of ratios
        red_ac, red_dc = MeerkatPipelineHelperfunctions.find_ac_signal(
            red_signal, self.intervall_length
        )
        ir_ac, ir_dc = MeerkatPipelineHelperfunctions.find_ac_signal(
            infrared_signal, self.intervall_length
        )

        delta_A_red = np.divide(red_ac, red_dc)
        delta_A_infrared = np.divide(ir_ac, ir_dc)[: len(delta_A_red)]
        delta_A_red = np.array(red_ac)
        delta_A_infrared = np.array(ir_ac)[: len(delta_A_red)]

        # Absorption parameters
        eta_red_hb = 3750.12
        eta_red_hbo2 = 368
        eta_infrared_hb = 691.3
        eta_infrared_hbo2 = 1058.0

        # Calculate final signal
        self.spO2_calibrated = (
            100
            * (eta_red_hb - eta_infrared_hb * (delta_A_red / delta_A_infrared))
            / (
                eta_red_hb
                - eta_red_hbo2
                + (eta_infrared_hbo2 - eta_infrared_hb)
                * (delta_A_red / delta_A_infrared)
            )
        )
        # Truncate excessively large and low values
        self.spO2_calibrated = self.clamp_spO2(self.spO2_calibrated)

        # Smooth
        self.spO2_calibrated = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_calibrated,
            self.process_noise,
            self.measurement_noise,
            self.initial_estimate_error,
        )

    def clamp_spO2(self, signal):
        """
        Clamps the values in self.spO2_ycgcr to be within the range [70, 100].
        """
        signal = np.clip(signal, 70, 100)
        return signal

    def plot_data(self):
        # Plot all calculated signals and ground truth
        colors = MeerkatPipelineHelperfunctions.set_plot_params()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(
            self.pulse_oximeter_timestamps,
            self.pulse_oximeter_spO2,
            label="Pulse oximeter",
            color=colors[0],
        )
        ax1.plot(self.time_spO2, self.spO2_infrared, label="Infrared", color=colors[1])
        ax1.plot(self.time_spO2, self.spO2_rgb, label="RGB", color=colors[2])

        self.set_spo2_plot_params(ax1)

        ax2.plot(
            self.pulse_oximeter_timestamps,
            self.pulse_oximeter_spO2,
            label="Pulse oximeter",
            color=colors[0],
        )
        ax2.plot(self.time_spO2, self.spO2_ycgcr, label="YCgCr", color=colors[1])
        ax2.plot(
            self.timestamps_spO2_calibrated,
            self.spO2_calibrated,
            label="Calibration free",
            color=colors[2],
        )
        self.set_spo2_plot_params(ax2)

        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def set_spo2_plot_params(self, ax):
        ax.set_ylabel("SpO2 (%)", fontsize=20)
        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_title("Infrared and RGB", fontsize=20)
        ax.set_ylim(68, 100)
        ax.legend(loc="lower right", fontsize=14, frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

    def statistical_analysis(self):
        """
        Perform statistical analysis on various signals and write intermediate files.
        """

        def run_analysis(vital_sign, reference_signal, reference_timestamps):
            """
            Helper function to set up and run statistical analysis.

            Args:
                vital_sign (str): The name of the vital sign being analyzed.
                reference_signal (array-like): The signal to be compared against the ground truth.
                reference_timestamps (array-like): The timestamps for the reference signal.
            """
            print(f"{vital_sign.replace('_', ' ').title()} Analysis")
            analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
            analysis.vital_sign = vital_sign
            analysis.subject_folder = self.subject_folder
            analysis.ground_truth_signal = self.pulse_oximeter_spO2
            analysis.ground_truth_timestamps = self.pulse_oximeter_timestamps
            analysis.reference_signal = reference_signal
            analysis.reference_timestamps = reference_timestamps
            analysis.kappa = 3
            analysis.run()

        # Perform analysis for each type of oxygen saturation
        run_analysis("ycgcr_oxygen_saturation", self.spO2_ycgcr, self.time_spO2)
        run_analysis("infrared_oxygen_saturation", self.spO2_infrared, self.time_spO2)
        run_analysis("rgb_oxygen_saturation", self.spO2_rgb, self.time_spO2)
        run_analysis(
            "calibration_free_oxygen_saturation", self.spO2_calibrated, self.time_spO2
        )

    def return_data(self):
        """
        Returns:
            dict: A dictionary with the following key-value pairs:
                - "Pulse Oximeter Timestamps" (array): Timestamps from the pulse oximeter.
                - "Pulse Oximeter SpO2" (array): SpO2 values from the pulse oximeter.
                - "Time SpO2" (array): Timestamps for SpO2 measurements.
                - "SpO2 Calibrated" (array): Calibrated SpO2 values.
                - "SpO2 RGB" (array): SpO2 values from RGB signal.
                - "SpO2 Infrared" (array): SpO2 values from Infrared signal.
                - "SpO2 YCgCr" (array): SpO2 values from YCgCr signal.
                - "Timestamps SpO2 Calibrated" (array): Timestamps for calibrated SpO2.
        """
        # Create a dictionary with all relevant data
        data = {
            "Pulse Oximeter Timestamps": self.pulse_oximeter_timestamps,
            "Pulse Oximeter SpO2": self.pulse_oximeter_spO2,
            "Time SpO2": self.time_spO2,
            "SpO2 Calibrated": self.spO2_calibrated,
            "SpO2 RGB": self.spO2_rgb,
            "SpO2 Infrared": self.spO2_infrared,
            "SpO2 YCgCr": self.spO2_ycgcr,
            "Timestamps SpO2 Calibrated": self.timestamps_spO2_calibrated,
        }

        return data
