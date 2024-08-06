import numpy as np
import matplotlib.pyplot as plt
import MeerkatPipelineHelperfunctions
import os
from scipy.signal import sosfiltfilt
import pandas as pd
from sklearn import decomposition
import StatisticalAnalysis
import sys
import seaborn as sns


class CalculatePulseOxygenation:
    def __init__(self):
        self.data_analysis_folder = ""
        self.intervall_length = 900

    def run(self):
        # Function to tie all functions in class together
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(
            self.data_analysis_folder
        )
        self.import_oxygen_saturation_data()
        self.import_oximeter_spO2()
        self.infrared_oxygenation()
        self.rgb_oxygenation()
        self.ycgcr_oxygenation()
        self.PCA_oxygenation()
        self.plot_data()
        self.statistical_analysis()

    def import_oxygen_saturation_data(self):
        # Define camera folder
        camera_folder = repr("\\" + "RGB-D camera video data")
        camera_folder = self.subject_folder + camera_folder[2:-1]
        camera_file = os.listdir(camera_folder)[0]
        camera_file = repr("\\" + camera_file)
        camera_filepath = camera_folder + camera_file[2:-1]

        # Read data from csv
        df = pd.read_csv(camera_filepath)
        self.ts1 = np.array(df["Time (s)"])
        self.red_signal = np.array(df[" Red"])
        self.green_signal = np.array(df[" Green"])
        self.blue_signal = np.array(df[" Blue"])
        self.infrared_signal = np.array(df[" IR"])

        # Calculate average signals
        self.mean_red = np.mean(self.red_signal)
        self.mean_blue = np.mean(self.blue_signal)
        self.mean_green = np.mean(self.green_signal)

        # Bandpass filter the signals
        self.ac_filter = MeerkatPipelineHelperfunctions.butter_bandpass(
            1, 5, 30, order=7
        )
        self.filtered_red = sosfiltfilt(self.ac_filter, self.red_signal) + np.mean(
            self.red_signal
        )
        self.filtered_blue = sosfiltfilt(self.ac_filter, self.blue_signal) + np.mean(
            self.blue_signal
        )
        self.filtered_ir = sosfiltfilt(self.ac_filter, self.infrared_signal) + np.mean(
            self.infrared_signal
        )
        self.filtered_green = sosfiltfilt(self.ac_filter, self.green_signal) + np.mean(
            self.infrared_signal
        )

    def import_oximeter_spO2(self):
        # Define pulse oximeter folder
        oximeter_folder = repr("\\" + "Pulse oximeter oxygen saturation")
        oximeter_folder = self.subject_folder + oximeter_folder[2:-1]
        if len(os.listdir(oximeter_folder)) > 0:  # if file exists import data from it
            oximeter_file = os.listdir(oximeter_folder)[0]
            oximeter_file = repr("\\" + oximeter_file)
            oximeter_filepath = oximeter_folder + oximeter_file[2:-1]
            df = pd.read_csv(oximeter_filepath)
            self.pulse_oximeter_timestamps = np.array(df["Time (s)"])
            self.pulse_oximeter_spO2 = np.array(df[" Oxygen saturation (%)"])
            self.pulse_oximeter_spO2 = MeerkatPipelineHelperfunctions.Kalman_filter(
                self.pulse_oximeter_spO2, 0.0001, 0.00001, 10
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

        # If values over 100 or below 75 truncate
        self.spO2_rgb = (
            40 * np.log(self.rr_signal_rgb) + 80
        )  # calibration can be adjusted as needed
        for i in range(len(self.spO2_rgb)):
            if self.spO2_rgb[i] > 100:
                self.spO2_rgb[i] = 100
            if self.spO2_rgb[i] < 70:
                self.spO2_rgb[i] = 70
        # Smooth using Kalman filter
        self.spO2_rgb = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_rgb, 0.0001, 0.00001, 10
        )

    def ycgcr_oxygenation(self):
        # Algorithm after https://www.mdpi.com/1424-8220/21/18/6120

        # Normalize data to 0:1 range
        r_normalized = np.array(self.red_signal) / 255
        g_normalized = np.array(self.green_signal) / 255
        b_normalized = np.array(self.blue_signal) / 255
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
        for i in range(len(self.rr_ycgcr)):
            if self.rr_ycgcr[i] < 0 or self.rr_ycgcr[i] > 1.8:
                if i > 0:
                    self.rr_ycgcr[i] = self.rr_ycgcr[i - 1]
                else:
                    self.rr_ycgcr[i] = 1
        # If values over 100 or below 75 truncate
        self.spO2_ycgcr = 11.88 * self.rr_ycgcr + 82  # calibration adjustable
        for i in range(len(self.spO2_ycgcr)):
            if self.spO2_ycgcr[i] > 100:
                self.spO2_ycgcr[i] = 100
            if self.spO2_ycgcr[i] < 70:
                self.spO2_ycgcr[i] = 70
        # Smooth using Kalman filter
        self.spO2_ycgcr = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_ycgcr, 0.0001, 0.00001, 10
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

        # Define timestamps for bins
        self.time_spO2 = np.linspace(
            self.ts1[self.intervall_length],
            self.ts1[-1],
            num=int((len((self.filtered_red)) - self.intervall_length) / 30),
        )

        # Calculate ratio of ratios
        red_signal = np.divide(red_ac_signal, red_dc_signal)
        ir_signal = np.divide(ir_ac_signal, ir_dc_signal)
        self.rr_signal_infrared = np.divide(red_signal, ir_signal)

        # Calculate SpO2 truncate at 100
        self.spO2_infrared = 100 + 20 / 6 - 20 / 3 * self.rr_signal_infrared
        for i in range(len(self.spO2_infrared)):
            if self.spO2_infrared[i] > 100:
                self.spO2_infrared[i] = 100
            if self.spO2_infrared[i] < 70:
                self.spO2_infrared[i] = 70
        self.spO2_infrared = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_infrared, 0.0001, 0.00001, 10
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
        for i in range(len(self.spO2_calibrated)):
            if self.spO2_calibrated[i] > 100:
                self.spO2_calibrated[i] = 100
            if self.spO2_calibrated[i] < 70:
                self.spO2_calibrated[i] = 70
        self.spO2_calibrated = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.spO2_calibrated, 0.0001, 0.00001, 10
        )
        self.timestamps_spO2_calibrated = np.linspace(
            self.ts1[self.intervall_length], self.ts1[-1], num=len(self.spO2_calibrated)
        )

    def plot_data(self):
        # Plot all calculated signals and ground truth
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
        plt.rcParams.update(params)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        colors = sns.color_palette("deep")
        ax1.plot(
            self.pulse_oximeter_timestamps,
            self.pulse_oximeter_spO2,
            label="Pulse oximeter",
            color=colors[0],
        )
        ax1.plot(self.time_spO2, self.spO2_infrared, label="Infrared", color=colors[1])
        ax1.plot(self.time_spO2, self.spO2_rgb, label="RGB", color=colors[2])
        ax1.set_ylabel("SpO2 (%)", fontsize=20)
        ax1.set_xlabel("Time (s)", fontsize=20)
        ax1.set_title("Infrared and RGB", fontsize=20)
        ax1.set_ylim(68, 100)
        ax1.legend(loc="lower right", fontsize=14, frameon=False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position("left")
        ax1.xaxis.set_ticks_position("bottom")
        ax1.tick_params(axis="x", labelsize=14)
        ax1.tick_params(axis="y", labelsize=14)

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
        ax2.set_ylabel("SpO2 (%)", fontsize=20)
        ax2.set_xlabel("Time (s)", fontsize=20)
        ax2.set_title("YCgCr and Calibration free", fontsize=20)
        ax2.set_ylim(68, 100)
        ax2.legend(loc="lower right", fontsize=14, frameon=False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position("left")
        ax2.xaxis.set_ticks_position("bottom")
        ax2.tick_params(axis="x", labelsize=14)
        ax2.tick_params(axis="y", labelsize=14)

        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def statistical_analysis(self):
        # Perform statistical analysis on all signals
        print("Oxygen saturation YCgCr")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "ycgcr_oxygen_saturation"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.pulse_oximeter_spO2
        Analysis.ground_truth_timestamps = self.pulse_oximeter_timestamps
        Analysis.reference_signal = self.spO2_ycgcr
        Analysis.reference_timestamps = self.time_spO2
        Analysis.kappa = 3
        Analysis.run()
        print("Oxygen saturation Infrared")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "infrared_oxygen_saturation"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.pulse_oximeter_spO2
        Analysis.ground_truth_timestamps = self.pulse_oximeter_timestamps
        Analysis.reference_signal = self.spO2_infrared
        Analysis.reference_timestamps = self.time_spO2
        Analysis.kappa = 3
        Analysis.run()
        print("Oxygen saturation RGB")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "rgb_oxygen_saturation"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.pulse_oximeter_spO2
        Analysis.ground_truth_timestamps = self.pulse_oximeter_timestamps
        Analysis.reference_signal = self.spO2_rgb
        Analysis.reference_timestamps = self.time_spO2
        Analysis.kappa = 3
        Analysis.run()
        print("Oxygen saturation calibration free")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "calibration_free_oxygen_saturation"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.pulse_oximeter_spO2
        Analysis.ground_truth_timestamps = self.pulse_oximeter_timestamps
        Analysis.reference_signal = self.spO2_calibrated
        Analysis.reference_timestamps = self.time_spO2
        Analysis.kappa = 3
        Analysis.run()
    def return_data(self):
        return self.pulse_oximeter_timestamps, self.pulse_oximeter_spO2, self.time_spO2, self.spO2_calibrated, self.spO2_rgb, self.spO2_infrared, self.spO2_ycgcr,self.timestamps_spO2_calibrated
