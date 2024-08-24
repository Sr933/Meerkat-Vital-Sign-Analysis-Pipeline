##### Meerkat vital sign pipeline #####

import MeerkatPipelineHelperfunctions
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks, sosfiltfilt
from scipy.signal.windows import hamming
from scipy.fft import fft, fftfreq
from scipy.stats import norm
import shutil
import csv
from sklearn import decomposition


class BreathingAsymmetryPipeline:
    def __init__(self, data_analysis_folder="", interval_length=1800):
        """
        Initialize the BreathingAsymmetry class.

        Parameters:
        - data_analysis_folder (str): Path to the folder containing data.
        - interval_length (int): Length of the sliding window interval in seconds.
        """
        self.data_analysis_folder = data_analysis_folder
        self.interval_length = interval_length

    def run(self):
        """
        Run the complete analysis process.
        """
        # Choose the subject folder using the helper function
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(
            self.data_analysis_folder
        )
        # Import respiratory data from camera
        self.import_camera_resp_data()
        # Calculate PCA signals for respiratory data
        self.calculate_pca_signals()
        # Find tidal volumes and peaks for both left and right sides
        self.find_tidal_volumes()
        # Calculate windowed volumes for both sides
        self.calculate_windowed_volumes()
        # Generate timestamps for the sliding windows
        self.generate_timestamps()
        # Calculate asymmetry scores
        self.calculate_asymmetry()
        # Plot the data
        self.plot_data()

    def import_camera_resp_data(self):
        """
        Import respiratory data from CSV files located in the subject folder.
        """
        # Path to the camera data folder
        camera_folder = os.path.join(self.subject_folder, "RGB-D camera video data")
        # Get the path to the first CSV file in the folder
        camera_filepath = os.path.join(camera_folder, os.listdir(camera_folder)[0])
        # Load the data into a DataFrame
        df = pd.read_csv(camera_filepath)

        # Extract relevant columns from the DataFrame
        self.ts1 = df["Time (s)"].to_numpy()
        self.ROI_x_1_array = df[" Rectangle x1"].to_numpy()
        self.ROI_x_2_array = df[" Rectangle x2"].to_numpy()
        self.ROI_y_1_array = df[" Rectangle y1"].to_numpy()
        self.ROI_y_2_array = df[" Rectangle y2"].to_numpy()
        self.left_chest_depth = df[" Depth Left Chest"].to_numpy()
        self.right_chest_depth = df[" Depth Right Chest"].to_numpy()

    def calculate_pca_signals(self):
        """
        Calculate PCA signals for the left and right chest respiratory data.
        """
        # Calculate PCA signals for left chest
        self.PCA_signal_left = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.left_chest_depth,
            self.ROI_x_1_array,
            self.ROI_x_2_array,
            self.ROI_y_1_array,
            self.ROI_y_2_array,
            outliers=False,
        )
        # Calculate PCA signals for right chest
        self.PCA_signal_right = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.right_chest_depth,
            self.ROI_x_1_array,
            self.ROI_x_2_array,
            self.ROI_y_1_array,
            self.ROI_y_2_array,
            outliers=False,
        )

    def find_tidal_volumes(self):
        """
        Find valid tidal volumes and peaks for the left and right sides.
        """
        # Find tidal volumes and peaks for left chest
        self.left_peaks, self.left_volumes, _, _ = (
            MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
                self.PCA_signal_left
            )
        )
        # Find tidal volumes and peaks for right chest
        self.right_peaks, self.right_volumes, _, _ = (
            MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
                self.PCA_signal_right
            )
        )

    def calculate_windowed_volumes(self):
        """
        Calculate average volumes in sliding windows for both left and right sides.
        """
        # Calculate volumes using sliding windows for left chest
        self.left_volumes_windowed = self.chest_half_movement_sliding_window(
            self.PCA_signal_left, self.left_peaks, self.left_volumes
        )
        # Calculate volumes using sliding windows for right chest
        self.right_volumes_windowed = self.chest_half_movement_sliding_window(
            self.PCA_signal_right, self.right_peaks, self.right_volumes
        )

    def chest_half_movement_sliding_window(
        self, signal, valid_peaks, valid_tidal_volumes
    ):
        """
        Calculate average tidal volumes in sliding windows.

        Parameters:
        - signal (array): PCA signal array.
        - valid_peaks (array): Array of valid peak indices.
        - valid_tidal_volumes (array): Array of valid tidal volumes.

        Returns:
        - interval_volumes (list): Average tidal volumes in sliding windows.
        """
        intervall_volumes = []

        valid_peaks = np.array(valid_peaks)

        interval_num=int((len((signal)) - self.interval_length) / 30)
        for i in range(interval_num):
            result = np.where(
                np.logical_and(
                    valid_peaks >= 30 * i, valid_peaks <= self.interval_length + 30 * i
                )
            )[0]
            n = len(result)
            if n > 0:
                volume_i = np.mean(valid_tidal_volumes[result])
            else:
                volume_i=intervall_volumes[-1] if len(intervall_volumes)>0 else 0.1
            intervall_volumes.append(volume_i)
        return intervall_volumes

    def generate_timestamps(self):
        """
        Generate timestamps for the sliding windows.
        """
        # Create an array of timestamps for the sliding windows
        self.window_timestamps = np.linspace(
            self.ts1[self.interval_length],
            self.ts1[-1],
            num=len(self.left_volumes_windowed),
        )
        
    def calculate_asymmetry(self):
        # Calculate the asymmetry score
        self.asymmetry_score = (
            200
            * (
                np.array(self.left_volumes_windowed)
                - np.array(self.right_volumes_windowed)
            )
            / (
                np.array(self.left_volumes_windowed)
                + np.array(self.right_volumes_windowed)
            )
        )

    def plot_data(self):
        """
        Plot the asymmetry score over time.
        """
        # Set plotting style and parameters
        colors = MeerkatPipelineHelperfunctions.set_plot_params()
        # Create a plot
        fig, ax = plt.subplots(figsize=(7.4, 4))
        ax.plot(self.window_timestamps, self.asymmetry_score, color=colors[0])
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Asymmetry score (%)", fontsize=14)
        ax.set_title("Breathing asymmetry", fontsize=14)

        # Customize plot appearance
        MeerkatPipelineHelperfunctions.plot_prettifier(ax)
        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def return_data(self):
        """
        Returns:
            dict: A dictionary containing:
                - 'Window timestamps' (array): An array of timestamps associated with each sliding window.
                - 'Asymmetry score' (array): An array of calculated asymmetry scores corresponding to the timestamps.
        """

        # Create a dictionary that maps descriptive keys to the respective data.
        data = {
            "Window timestamps": self.window_timestamps,  # Timestamps for each sliding window.
            "Asymmetry score": self.asymmetry_score,  # Asymmetry scores for the respective windows.
        }
        return data









class FlowVolumeLoopPipeline:
    def __init__(self, data_analysis_folder="", hasventilator=False, plot_single_loops_flag=False):
        """
        Initializes the CalculateFlowVolumeLoop class with the provided parameters.

        Args:
            data_analysis_folder (str): Path to the data analysis folder.
            hasventilator (bool): Flag indicating the presence of ventilator data.
            plot_single_loops_flag (bool): Flag to enable or disable plotting individual loops.
        """
        # Path to the data analysis folder
        self.data_analysis_folder = data_analysis_folder
        
        # Flag indicating the presence of ventilator data
        self.hasventilator = hasventilator
        
        # Flag for plotting individual loops
        self.plot_single_loops_flag = plot_single_loops_flag
        
        # Placeholder for the path to the intermediate file
        self.intermediate_file = None
        
        # Placeholder for the path to the intermediate folder
        self.intermediate_folder = None


    def run_part1(self):
        # Select subject folder and preprocess data if ventilator data is available
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(
            self.data_analysis_folder
        )
        self.preprocess_data_intermediate()
        return (self.intermediate_file, True) if self.hasventilator else (None, False)

    def run_part2(self):
        # Continue preprocessing after running ventiliser package and plotting data
        self.preprocess_data()

        if self.hasventilator:
            self.import_ventiliser_results()
        if self.plot_single_loops_flag and self.hasventilator:
            self.plot_single_loops()
        else:
            self.plot_data()

    def preprocess_data_intermediate(self):
        # Import respiratory data and ventilator flow data, apply filtering if ventilator data is available
        self.import_camera_resp_data()
        self.import_ventilator_flow()
        if self.hasventilator:
            butter = MeerkatPipelineHelperfunctions.butter_bandpass(
                15 / 60, 150 / 60, 100, order=7
            )
            self.ventilator_flow = sosfiltfilt(butter, self.ventilator_flow)
            self.ventilator_pressure = sosfiltfilt(butter, self.ventilator_pressure)
            self.write_flow_pressure_intermediate_file()

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

    def import_ventilator_flow(self):
        # Load ventilator flow data if available
        ventilator_flow_folder = os.path.join(self.subject_folder, "Ventilator flow")
        if os.listdir(ventilator_flow_folder):
            self.hasventilator = True
            ventilator_flow_file = os.listdir(ventilator_flow_folder)[0]
            ventilator_filepath = os.path.join(
                ventilator_flow_folder, ventilator_flow_file
            )
            df = pd.read_csv(ventilator_filepath)
            self.ventilator_flow = df[" Flow (l/min)"].to_numpy()
            self.ventilator_time = df["Time (s)"].to_numpy()
            self.ventilator_pressure = df[" Pressure (mbar)"].to_numpy()
        else:
            print("No ventilator data")

    def write_flow_pressure_intermediate_file(self):
        # Write filtered ventilator flow and pressure data to an intermediate file
        self.intermediate_folder = os.path.join(
            self.data_analysis_folder, "pipeline intermediate"
        )
        if os.path.exists(self.intermediate_folder):
            shutil.rmtree(self.intermediate_folder)
        os.makedirs(self.intermediate_folder)
        
        self.intermediate_file = os.path.join(
            self.intermediate_folder, "pipeline intermediate.csv"
        )
        
        # Write file
        with open(self.intermediate_file, "w", newline="") as file:
            writer = csv.writer(file)
            for t, p, f in zip(
                self.ventilator_time, self.ventilator_pressure, self.ventilator_flow
            ):
                writer.writerow([t, p, f])

    def preprocess_data(self):
        """
        Preprocesses the respiratory data by performing PCA on depth signals and calculating the flow.

        This function processes depth signals from different regions, applies PCA to obtain a combined 
        signal, calculates the flow by numerical differentiation, and identifies valid tidal volumes.
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

        # Calculate flow by numerical differentiation
        self.calculate_flow()
        
        # Identify valid tidal volumes and related statistics
        self.valid_peaks, self.valid_tidal_volumes, self.mean, self.std = (
            MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
                self.PCA_signal_vol
            )
        )


    def calculate_flow(self):
        # Calculate flow from the PCA signal
        self.flow = np.diff(self.PCA_signal_vol) / np.diff(self.ts1)

    def import_ventiliser_results(self):
        # Import ventilator analysis results
        ventiliser_file = os.listdir(self.intermediate_folder)[2]
        ventiliser_filepath = os.path.join(self.intermediate_folder, ventiliser_file)
        with open(ventiliser_filepath) as f:
            lines = f.readlines()[1:]
            self.breath_start, self.breath_end = zip(
                *[line.split(",")[1:3] for line in lines]
            )

    def valid_loop(self, breath_flow, breath_volume, flow_crossings):
        # Determine if a given flow volume loop meets physiological criteria
        conditions = (
            abs(breath_flow[-1]) < 20,
            abs(breath_flow[0]) < 20,
            abs(breath_volume[-1]) < 1,
            self.mean - self.std < max(np.absolute(breath_volume)) < self.mean + self.std,
            flow_crossings < 3,
            max(np.abs(breath_flow)) < 40,
            min(breath_volume) < -2
        )

        return all(conditions)


    def camera_loop_calc(self, j):
        # Calculate flow-volume loop from camera data
        breath_volume = [0]
        breath_flow = np.array(self.flow[self.valid_peaks[j] : self.valid_peaks[j + 1]])
        breath_volume = (
            np.array(self.PCA_signal_vol[self.valid_peaks[j] : self.valid_peaks[j + 1]])
            - self.PCA_signal_vol[self.valid_peaks[j]]
        )
        flow_crossings = ((breath_flow[:-1] * breath_flow[1:]) < 0).sum()
        return breath_volume, breath_flow, flow_crossings
    
    def ventilator_loop_calc(self, j):
        # Calculate flow-volume loop from ventilator data
        breath_volume = [0]
        breath_flow = (
            -np.array(
                self.ventilator_flow[
                    int(self.breath_start[j]) : int(self.breath_end[j])
                ]
            )
            * 1000
            / 60
        ) # Unit conversion needed
        breath_time = np.array(
            self.ventilator_time[
                int(self.breath_start[j]) : int(self.breath_end[j])
            ]
        )
            
        flow_crossings = ((breath_flow[:-1] * breath_flow[1:]) < 0).sum()
        
        # Calculate time differences between consecutive elements
        time_diff = np.diff(breath_time)

        # Calculate the average flow between consecutive elements
        avg_flow = 0.5 * (breath_flow[:-1] + breath_flow[1:])

        # Compute the incremental volumes
        incremental_volume = time_diff * avg_flow

        # Accumulate the incremental volumes to get the total volume at each step
        breath_volume = np.concatenate(([breath_volume[0]], np.cumsum(incremental_volume) + breath_volume[0]))
      
        return breath_volume, breath_flow, flow_crossings

    def plot_data(self):
        # Store loops
        self.camera_loops_flow = []
        self.camera_loops_volume = []
        self.ventilator_loops_flow = []
        self.ventilator_loops_volume = []
        # Plot flow-volume loops depending on whether ventilator loops are available

        colors = MeerkatPipelineHelperfunctions.set_plot_params()
        if self.hasventilator:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 6))

        # Iterate over peaks and calculate flow-volume loop for each peak
        breath_number=len(self.valid_peaks) - 1
        for j in range(breath_number):
            breath_volume, breath_flow, flow_crossings = self.camera_loop_calc(j)
            # Criteria for fv loop visualisation
            if self.valid_loop(breath_flow, breath_volume, flow_crossings):
                ax1.plot(breath_volume, breath_flow, color=colors[j % 10])
                self.camera_loops_volume.append(breath_volume)
                self.camera_loops_flow.append(breath_flow)

        # Define plot parameters
        self.set_fv_plot_params(ax1)
        ax1.set_title("Camera", fontsize=20)

        # Calculate ventilator loop
        if self.hasventilator:
            # Iterate over peaks and calculate flow volume loop
            ventilator_breath_number=len(self.breath_start) - 1
            for j in range(ventilator_breath_number):
                    breath_volume, breath_flow, flow_crossings = self.ventilator_loop_calc(j)
                    # Criteria for fv loop visualisation
                    if self.valid_loop(
                        breath_flow, breath_volume, flow_crossings
                    ):
                        ax2.plot(breath_volume, breath_flow, color=colors[j % 10])
                        self.ventilator_loops_volume.append(breath_volume)
                        self.ventilator_loops_flow.append(breath_flow)

            # Define plot parameters
            self.set_fv_plot_params(ax2)
            ax2.set_title("Ventilator", fontsize=20)
        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def plot_single_loops(self):
        # Plot parameters
        colors = MeerkatPipelineHelperfunctions.set_plot_params()
        camera_breath_number=len(self.valid_peaks)-1
        for i in range(camera_breath_number):
            camera_peak_t = self.ts1[i]
            ventilator_breath_number=len(self.breath_start) - 1
            for j in range(ventilator_breath_number):
                ventilator_peak_t = self.ventilator_time[int(self.breath_start[j])]
                if abs(camera_peak_t - ventilator_peak_t) < 2.0:

                    breath_volume_camera, breath_flow_camera, flow_crossings_camera = self.camera_loop_calc(i)
                    breath_volume_ventilator, breath_flow_ventilator, flow_crossings_ventilator = self.ventilator_loop_calc(j)
                    
                    # Define plot parameters
            
                    # Show loops if both loops are considered valid
                    if self.valid_loop(breath_flow_camera, breath_volume_camera, flow_crossings_camera) and self.valid_loop(breath_flow_ventilator, breath_volume_ventilator, flow_crossings_ventilator):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                        # Define plot parameters
                        self.set_fv_plot_params(ax1)
                        ax1.set_title("Camera", fontsize=20)
                        self.set_fv_plot_params(ax2)
                        ax2.set_title("Ventilator", fontsize=20)
                        
                        ax1.plot(breath_volume_camera, breath_flow_camera, color=colors[0])
                        ax2.plot(breath_volume_ventilator, breath_flow_ventilator, color=colors[0])
                        plt.tight_layout(pad=2.5, w_pad=2.5)
                        plt.show()

    def set_fv_plot_params(self, ax):
        ax.set_xlabel("Volume (ml)", fontsize=20)
        ax.set_ylabel("Flow (ml/s)", fontsize=20)
        ax.set_xlim(-10, 1)
        ax.set_ylim(-60, 60)
        ax.grid()
        MeerkatPipelineHelperfunctions.plot_prettifier(ax)

    def return_data(self):
        """
        Returns:
            dict: A dictionary containing the following key-value pairs:
                - "Ventilator loops flow" (array): Flow data from the ventilator loops.
                - "Ventilator loops volume" (array): Volume data from the ventilator loops.
                - "Camera loops flow" (array): Flow data from the camera loops.
                - "Camera loops volume" (array): Volume data from the camera loops.
        """
        
        # Create a dictionary to store the flow and volume data for both ventilator and camera loops.
        data = {
            "Ventilator loops flow": self.ventilator_loops_flow,   # Flow data from ventilator loops.
            "Ventilator loops volume": self.ventilator_loops_volume, # Volume data from ventilator loops.
            "Camera loops flow": self.camera_loops_flow,           # Flow data from camera loops.
            "Camera loops volume": self.camera_loops_volume,       # Volume data from camera loops.
        }
        return data




class HeartRatePipeline:
    def __init__(self, data_analysis_folder="", interval_length=3600):
        """
        Initialize the CalculateHeartRate class with default parameters.

        Args:
            intervall_length (int): Length of the interval in seconds.
            data_analysis_folder (str): Path to the folder containing data to analyze.
        """
        self.interval_length = interval_length
        self.data_analysis_folder = data_analysis_folder

    def run(self):
        # Function to tie all functions in class together
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(
            self.data_analysis_folder
        )
        self.load_and_process_data()
        self.plot_data()
        self.statistical_analysis()

    def load_and_process_data(self):
        # Import camera and ECG data
        self.import_heart_camera_data()
        self.import_ECG_rate()

        # Calculate heart rate using POS and CHROM
        self.butter = MeerkatPipelineHelperfunctions.butter_bandpass(
            1.5, 4.5, 30, order=7
        )
        self.CHROM_signal()
        self.POS_signal()

        # Calculate heart rate from POS and CHROM signals using peak counting
        self.heart_rate_CHROM = self.find_heart_rate(
            self.heart_rate_signal_CHROM
        )
        self.heart_rate_POS = self.find_heart_rate(
            self.heart_rate_signal_POS_filtered
        )
        self.generate_timestamps(self.heart_rate_signal_CHROM)
        
        # Calculate heart rate from POS and CHROM signals using Fourier analysis with Bayesian Inference
        self.CHROM_fourier = self.fourier_heart_rate(self.heart_rate_signal_CHROM)
        self.POS_fourier = self.fourier_heart_rate(self.heart_rate_signal_POS)

        # Define the Kalman filter parameters
        process_noise = 0.0001
        measurement_noise = 0.00001
        initial_estimate_error = 25

        # Define the signals and their corresponding attributes
        self.signals = {
            "CHROM_kalman_peaks": self.heart_rate_CHROM,
            "POS_kalman_peaks": self.heart_rate_POS,
            "CHROM_kalman_fourier": self.CHROM_fourier,
            "POS_kalman_fourier": self.POS_fourier,
            "ECG_kalman": self.ECG_rate
        }

        # Apply the Kalman filter to each signal and assign the result to the corresponding attribute
        for signal in self.signals:
            self.signals[signal]=MeerkatPipelineHelperfunctions.Kalman_filter(self.signals[signal], process_noise, measurement_noise, initial_estimate_error)


    def import_ECG_rate(self):
        # Define ECG file folder
        hr_folder = os.path.join(self.subject_folder, "ECG heart rate")
 
        if os.listdir(hr_folder):  # if file exists import data from it
            hr_file = os.path.join(hr_folder, os.listdir(hr_folder)[0])
            # Load data from csv
            df = pd.read_csv(hr_file)
            self.ECG_timestamps = df["Time (s)"].to_numpy()
            self.ECG_rate = df[" Heart rate (bpm)"].to_numpy()
        else:
            print("No valid heart rate data")
            sys.exit()

    def import_heart_camera_data(self):
        # Define camera filepath
        camera_folder = os.path.join(self.subject_folder, "RGB-D camera video data")
        camera_file=os.path.join(camera_folder, os.listdir(camera_folder)[0])
        
        # Load data from csv file
        df = pd.read_csv(camera_file)
        self.ts1 = df["Time (s)"].to_numpy()
        self.red_signal = df[" Red"].to_numpy()
        self.green_signal = df[" Green"].to_numpy()
        self.blue_signal = df[" Blue"].to_numpy()
        
        #Calculate mean values
        self.mean_red = np.mean(self.red_signal)
        self.mean_blue = np.mean(self.blue_signal)
        self.mean_green = np.mean(self.green_signal)

    def CHROM_signal(self):
        """
        Compute the CHROM signal using the algorithm from the reference:
        https://ieeexplore.ieee.org/document/6523142

        The method processes red, green, and blue signals to compute a heart rate signal
        based on the CHROM algorithm. It involves standardizing color channels, 
        constructing orthogonal channels, filtering, and calculating the final signal.
        """

        # Define standardised colour channel
        R_s = 0.7682 / self.mean_red * self.red_signal
        B_s = 0.5121 / self.mean_blue * self.blue_signal
        G_s = 0.3841 / self.mean_green * self.green_signal

        # Construct orthogonal colour channels
        X = (R_s - G_s) / (0.7672 - 0.5121)
        Y = (R_s + G_s - 2 * B_s) / (0.7682 + 0.5121 - 0.7682)

        # Bandpass filter orthogonal channels
        X_f = sosfiltfilt(self.butter, X)
        Y_f = sosfiltfilt(self.butter, Y)
        
        # Calculate the alpha coefficient
        alpha = np.std(X_f) / np.std(Y_f)

        # Filter original colour channels
        R_f = sosfiltfilt(self.butter, self.red_signal)
        G_f = sosfiltfilt(self.butter, self.green_signal)
        B_f = sosfiltfilt(self.butter, self.blue_signal)

        # Calculate final signal
        self.heart_rate_signal_CHROM = (
            3 * (1 - alpha / 2) * R_f
            - 2 * (1 + alpha / 2) * G_f
            + 3 * alpha / 2 * B_f.T
        )

    def POS_signal(self):
        """
        Compute the POS signal using the algorithm described in:
        https://ieeexplore.ieee.org/document/7565547

        """
        n = len(self.red_signal)  # signal length
        self.heart_rate_signal_POS = np.zeros(n)
        l = 48  # window length

        # Iterate over signal and calculate
        for i in range(n):
            m = i - l + 1
            if m > 0:
                # Get window signal
                red = self.red_signal[m:i]
                green = self.green_signal[m:i]
                blue = self.blue_signal[m:i]

                # Calculate means
                red = red - np.mean(red)
                green = green - np.mean(green)
                blue = blue - np.mean(blue)

                # Define orthogonal colour channels
                signal_1 = green - blue
                signal_2 = green + blue - 2 * red
                h = signal_1 + np.std(signal_1) / np.std(signal_2) * signal_2
                # Iterate over window
                for j in range(len(h)):
                    self.heart_rate_signal_POS[m + j] = (
                        self.heart_rate_signal_POS[m + j] + h[j] - np.mean(h)
                    )
        # Filter final signal
        self.heart_rate_signal_POS_filtered = sosfiltfilt(
            self.butter, self.heart_rate_signal_POS)

    def find_heart_rate(self, heart_rate_signal):
        # Find peaks in signal
        peaks, _ = find_peaks(heart_rate_signal, distance=6, height=0)
    
        # Define the number of intervals
        num_intervals = int((len(heart_rate_signal) - self.interval_length) / 30)

        # Initialize array for interval peak numbers
        intervall_peak_numbers = np.zeros(num_intervals)

        # Compute the start and end indices for each interval
        interval_starts = 30 * np.arange(num_intervals)
        interval_ends = self.interval_length + 30 * np.arange(num_intervals)

        # Process each interval
        for idx in range(num_intervals):
            start = interval_starts[idx]
            end = interval_ends[idx]

            # Find peaks within the current interval
            mask = (peaks >= start) & (peaks <= end)
            interval_peaks = peaks[mask]
            
            if len(interval_peaks) > 1:
                first_peak = interval_peaks[0]
                last_peak = interval_peaks[-1]
                peak_num= (len(interval_peaks) - 1) * 60 / (last_peak - first_peak) * 30
       
            else:
                peak_num=interval_peaks[idx-1] if idx>0 else 0
            intervall_peak_numbers[idx]=peak_num #default value
        return intervall_peak_numbers
    
    def generate_timestamps(self, heart_rate_signal):
        # Calculate timestamps of intervalls
        self.timestamps = np.linspace(
            self.ts1[self.interval_length],
            self.ts1[-1],
            num=int((len((heart_rate_signal)) - self.interval_length) / 30),
        )
       

    def fourier_heart_rate(self, signal):
        # Preprocessing of data using bandpass filter and timeseries PCA

        
        num_intervals = int((len(signal) - self.interval_length) / 30)
        fourier_rate = np.zeros(num_intervals)
        for i in range(num_intervals):
            # Calculate window signal
            window_signal = signal[30 * i : 30 * i + self.interval_length]
            n = len(window_signal)
            t = np.arange(n)
            w = hamming(n)
            
            # Perfor frequency analysis
            yf = np.abs(fft(window_signal * w))
            xf = fftfreq(t.shape[-1], 1 / 1800)

            # Perform inference on resp rate using prior of expected frequency and likelihood of Fourier coefficient
            mean = 155
            std = 15
            
            resp_gaussian = norm.pdf(xf, mean, std)
            yf = np.multiply(yf, resp_gaussian)
            fourier_rate[i]=xf[np.where(yf == np.max(yf))[0][0]]
        return fourier_rate

    def plot_data(self):
        """
        Plot the obtained data in comparison to the ground truth ECG.

        This method generates two subplots:
        1. Peak HR of POS and CHROM signals against ECG.
        2. Fourier HR of POS and CHROM signals, along with their average, against ECG.
        """
        
        # Calculate average of CHROM and POS
        self.average = (self.signals["CHROM_kalman_fourier"] + self.signals["POS_kalman_fourier"]) / 2

        # Define plot parameters
        colors = MeerkatPipelineHelperfunctions.set_plot_params()

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        # Plot data on the first subplot
        self.plot_peaks_hr(ax1, colors)

        # Plot data on the second subplot
        self.plot_fourier_hr(ax2, colors)

        # Adjust layout and display the plot
        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def plot_peaks_hr(self, ax, colors):
        """
        Plot the peak estimated HR of POS and CHROM signals against the ECG data on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
            colors (list): List of colors for the plot.
        """
        ax.plot(self.ECG_timestamps, self.signals["ECG_kalman"], label="ECG", color=colors[0])
        ax.plot(self.timestamps, self.signals["POS_kalman_peaks"], label="Peak POS", color=colors[1])
        ax.plot(self.timestamps, self.signals["CHROM_kalman_peaks"], label="Peak CHROM", color=colors[2])
        ax.legend(loc="lower center", fontsize=14, frameon=False, ncol=3)
        ax.set_title("Peak counting", fontsize=20)
        self.set_hr_plot_params(ax)

    def plot_fourier_hr(self, ax, colors):
        """
        Plot the Fourier estimated HR from POS and CHROM signals and their average against the ECG data on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
            colors (list): List of colors for the plot.
        """
        ax.plot(self.ECG_timestamps, self.signals["ECG_kalman"], label="ECG", color=colors[0])
        ax.plot(self.timestamps, self.signals["POS_kalman_fourier"], label="Fourier POS", color=colors[1])
        ax.plot(self.timestamps, self.signals["CHROM_kalman_fourier"], label="Fourier CHROM", color=colors[2])
        ax.plot(self.timestamps, self.average, label="POS CHROM Average", color=colors[3])
        ax.legend(loc="lower center", fontsize=14, frameon=False, ncol=2)
        ax.set_title("Fourier", fontsize=20)
        self.set_hr_plot_params(ax)

            
    def set_hr_plot_params(self, ax):
        ax.set_ylabel("Heart rate (bpm)", fontsize=20)
        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_ylim(100, 200)
        MeerkatPipelineHelperfunctions.plot_prettifier(ax)
        

    def statistical_analysis(self):
        """
        Perform statistical analysis for each signal and write intermediate files.
        """
        
        def run_analysis(vital_sign, reference_signal):
            """
            Helper function to configure and run the statistical analysis.
            
            Args:
                vital_sign (str): Description of the vital sign for the analysis.
                reference_signal (array): The reference signal for the analysis.
            """
            print(f"{vital_sign} analysis")
            analysis = MeerkatStatisticalAnalysis()
            analysis.vital_sign = vital_sign
            analysis.subject_folder = self.subject_folder
            analysis.ground_truth_signal = self.signals["ECG_kalman"]
            analysis.ground_truth_timestamps = self.ECG_timestamps
            analysis.reference_signal = reference_signal
            analysis.reference_timestamps = self.timestamps
            analysis.run()
        
        # Run the statistical analyses
        run_analysis("POS_Peak_counting", self.signals["POS_kalman_peaks"])
        run_analysis("CHROM_Peak_counting", self.signals["CHROM_kalman_peaks"])
        run_analysis("POS_Fourier_analysis", self.signals["POS_kalman_fourier"])
        run_analysis("CHROM_Fourier_analysis", self.signals["CHROM_kalman_fourier"])
        run_analysis("CHROM_POS", self.average)


    def return_data(self):
        """
        Return a dictionary containing various data attributes related to ECG and signal processing.

        Returns:
            dict: A dictionary with the following key-value pairs:
                - "ECG timestamps" (array): Timestamps corresponding to the ECG data.
                - "ECG rate" (array): ECG rate data.
                - "POS kalman" (array): Processed POS signal data after Kalman filtering.
                - "CHROM kalman" (array): Processed CHROM signal data after Kalman filtering.
                - "POS kalman peaks" (array): Detected peaks in the POS signal.
                - "CHROM kalman peaks" (array): Detected peaks in the CHROM signal.
                - "Timestamps" (array): General timestamps for the data.
                - "Average" (array): Average of the POS and CHROM signals.
        """
        
        # Create a dictionary to store the data attributes
        data = {
            "ECG timestamps": self.ECG_timestamps,       # Timestamps corresponding to the ECG data
            "ECG rate": self.signals["ECG_kalman"],                   # ECG rate data
            "POS kalman fourier": self.signals["POS_kalman__fourier"],               # Processed POS signal data
            "CHROM kalman fourier": self.signals["CHROM_kalman_fourier"],           # Processed CHROM signal data
            "POS kalman peaks": self.signals["POS_kalman_peaks"],   # Detected peaks in the POS signal
            "CHROM kalman peaks": self.signals["CHROM_kalman_peaks"], # Detected peaks in the CHROM signal
            "Timestamps": self.timestamps,               # General timestamps for the data
            "Average": self.average                      # Average of the POS and CHROM signals
        }
        
        return data



class PulseOxygenationPipeline:
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
        self.process_noise = 0.00001
        self.measurement_noise = 0.000001
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
            self.filtered_red, self.interval_length
        )
        blue_ac_signal, blue_dc_signal = MeerkatPipelineHelperfunctions.find_ac_signal(
            self.filtered_blue, self.interval_length
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
            filtered_cg, self.interval_length
        )
        cr_ac_signal, _ = MeerkatPipelineHelperfunctions.find_ac_signal(
            filtered_cr, self.interval_length
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
            self.filtered_red, self.interval_length
        )
        ir_ac_signal, ir_dc_signal = MeerkatPipelineHelperfunctions.find_ac_signal(
            self.filtered_ir, self.interval_length
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
            self.ts1[self.interval_length],
            self.ts1[-1],
            num=int((len((self.filtered_red)) - self.interval_length) / 30),
        )
        self.timestamps_spO2_calibrated = np.linspace(
            self.ts1[self.interval_length], self.ts1[-1], num=len(self.spO2_calibrated)
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
            red_signal, self.interval_length
        )
        ir_ac, ir_dc = MeerkatPipelineHelperfunctions.find_ac_signal(
            infrared_signal, self.interval_length
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
        MeerkatPipelineHelperfunctions.plot_prettifier(ax)

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
            analysis = MeerkatStatisticalAnalysis()
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




class MeerkatStatisticalAnalysis:
    def __init__(self):
            """
            Initialize the MeerkatStatisticalAnalysis class with optional parameters.
            
            Parameters:
            - ground_truth_signal: List of ground truth signal values.
            - reference_signal: List of reference signal values.
            - ground_truth_timestamps: List of timestamps for ground truth signals.
            - reference_timestamps: List of timestamps for reference signals.
            - kappa: Sensitivity parameter for statistical analysis.
            - subject_folder: Directory path for saving results.
            - vital_sign: Type of vital sign being analyzed.
            - tidalvolumeflag: Flag indicating if tidal volume data is included.
            - tidalvolumeupper: List of upper bounds for tidal volume.
            - tidalvolumelower: List of lower bounds for tidal volume.
            """
            self.ground_truth_signal = []
            self.reference_signal = []
            self.ground_truth_timestamps = []
            self.reference_timestamps = []
            self.kappa = 10
            self.subject_folder = ""
            self.vital_sign = ""
            self.tidalvolumeflag = False
            self.tidalvolumeupper = []
            self.tidalvolumelower = []

    def run(self):
        # Function to tie all functions in class together

        if self.tidalvolumeflag:
            self.ground_truth_signal = self.tidalvolumeupper
            self.match_signals()
            self.tidalvolumeupper_matched = self.matched_ground_truth_signal.copy()
            self.ground_truth_signal = self.tidalvolumelower
            self.match_signals()
            self.tidalvolumelower_matched = self.matched_ground_truth_signal.copy()
            self.tidal_volume_matching()
            self.matched_ground_truth_signal = self.truth_upper_lower

        else:
            self.match_signals()
        self.save_matched_signals()
        MeerkatPipelineHelperfunctions.mean_absolute_diff(
            self.matched_ground_truth_signal, self.matched_reference_signal
        )
        MeerkatPipelineHelperfunctions.mean_square_diff(
            self.matched_ground_truth_signal, self.matched_reference_signal
        )
        MeerkatPipelineHelperfunctions.coverage_probability(
            self.matched_ground_truth_signal, self.matched_reference_signal, self.kappa
        )
        self.kappa = 2 * self.kappa
        MeerkatPipelineHelperfunctions.coverage_probability(
            self.matched_ground_truth_signal, self.matched_reference_signal, self.kappa
        )

    def match_signals(self):
        # match the signals sampled at different rates with small ofsetts by caculating averages in the higher sampled signal
        ground_truth_signal = np.array(self.ground_truth_signal)
        reference_signal = np.array(self.reference_signal)
        ground_truth_timestamps = np.array(self.ground_truth_timestamps)
        reference_timestamps = np.array(self.reference_timestamps)
        matched_ground_truth = []
        matched_reference = []
        if len(ground_truth_signal) >= len(reference_signal):
            for i in range(len(reference_signal) - 1):
                timestart = reference_timestamps[i]
                timeend = reference_timestamps[i + 1]
                value_indices_in_window = np.where(
                    np.logical_and(
                        ground_truth_timestamps >= timestart,
                        ground_truth_timestamps <= timeend,
                    )
                )[0]

                values_in_window = []
                if len(value_indices_in_window > 0):
                    for j in value_indices_in_window:
                        if j < len(ground_truth_signal):
                            values_in_window.append(ground_truth_signal[j])

                    matched_ground_truth.append(np.mean(np.array(values_in_window)))
                    matched_reference.append(
                        (reference_signal[i])
                    )  

        else :
            for i in range(len(ground_truth_signal) - 1):
                timestart = ground_truth_timestamps[i]
                timeend = ground_truth_timestamps[i + 1]
                value_indices_in_window = np.where(
                    np.logical_and(
                        reference_timestamps >= timestart,
                        reference_timestamps <= timeend,
                    )
                )[0]

                values_in_window = []
                if len(value_indices_in_window) > 0:
                    for j in value_indices_in_window:
                        if j < len(reference_signal):
                            values_in_window.append(reference_signal[j])
                    matched_reference.append(np.mean(np.array(values_in_window)))
                    matched_ground_truth.append(ground_truth_signal[i])

        self.matched_ground_truth_signal = np.array(matched_ground_truth)
        self.matched_reference_signal = np.array(matched_reference)
        print("Data points:", len(self.matched_ground_truth_signal))

    def save_matched_signals(self):
        output_folder = os.path.join(self.subject_folder, "Pipeline results")
        output_filepath = os.path.join(output_folder, self.vital_sign)
        
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        
        with open(output_filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Ground Truth Signal', 'Reference Signal'])
            
            # Write the data rows
            for gt_signal, ref_signal in zip(self.matched_ground_truth_signal, self.matched_reference_signal):
                writer.writerow([gt_signal, ref_signal])

    def tidal_volume_matching(self):
        self.truth_upper_lower = np.zeros(len(self.matched_reference_signal))
        for i in range(len(self.matched_reference_signal)):
            if (
                self.matched_reference_signal[i] >= self.tidalvolumelower_matched[i]
                and self.matched_reference_signal[i] <= self.tidalvolumeupper_matched[i]
            ):
                self.truth_upper_lower[i] = self.matched_reference_signal[i]
            else:
                upper_bound = abs(
                    self.matched_reference_signal[i] - self.tidalvolumeupper_matched[i]
                )
                lower_bound = abs(
                    self.matched_reference_signal[i] - self.tidalvolumelower_matched[i]
                )
                if upper_bound > lower_bound:
                    self.truth_upper_lower[i] = self.tidalvolumelower_matched[i]
                else:
                    self.truth_upper_lower[i] = self.tidalvolumeupper_matched[i]





class RespiratoryPipeline:
    def __init__(
        self,
        data_analysis_folder="",
        intervall_length_resp=1800,
        intervall_length_vol=1800
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
        ax2.set_ylim(15, 80)
        self.resp_plot_params(ax2)

        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def resp_plot_params(self, ax):
        MeerkatPipelineHelperfunctions.plot_prettifier(ax)
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
            analysis = MeerkatStatisticalAnalysis()
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
