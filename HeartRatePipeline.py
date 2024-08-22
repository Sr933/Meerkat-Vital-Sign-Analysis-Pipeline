import MeerkatPipelineHelperfunctions
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, sosfiltfilt
from scipy.stats import norm
from scipy.signal.windows import hamming
from scipy.fft import fft, fftfreq
import StatisticalAnalysis
import sys


class Calculate_heart_rate:
    def __init__(self, intervall_length=3600, data_analysis_folder=""):
        """
        Initialize the CalculateHeartRate class with default parameters.

        Args:
            intervall_length (int): Length of the interval in seconds.
            data_analysis_folder (str): Path to the folder containing data to analyze.
        """
        self.intervall_length = intervall_length
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
        initial_estimate_error = 20

        # Define the signals and their corresponding attributes
        signals = [
            ("CHROM_kalman_peaks", self.heart_rate_CHROM),
            ("POS_kalman_peaks", self.heart_rate_POS),
            ("CHROM_kalman", self.CHROM_fourier),
            ("POS_kalman", self.POS_fourier),
            ("ECG_rate", self.ECG_rate),
        ]

        # Apply the Kalman filter to each signal and assign the result to the corresponding attribute
        for attr, signal in signals:
            setattr(self, attr, MeerkatPipelineHelperfunctions.Kalman_filter(signal, process_noise, measurement_noise, initial_estimate_error))


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
        num_intervals = int((len(heart_rate_signal) - self.intervall_length) / 30)

        # Initialize array for interval peak numbers
        intervall_peak_numbers = np.zeros(num_intervals)

        # Compute the start and end indices for each interval
        interval_starts = 30 * np.arange(num_intervals)
        interval_ends = self.intervall_length + 30 * np.arange(num_intervals)

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
                intervall_peak_numbers[idx] = (len(interval_peaks) - 1) * 60 / (last_peak - first_peak) * 30
        
        return intervall_peak_numbers
    
    def generate_timestamps(self, heart_rate_signal):
        # Calculate timestamps of intervalls
        self.timestamps = np.linspace(
            self.ts1[self.intervall_length],
            self.ts1[-1],
            num=int((len((heart_rate_signal)) - self.intervall_length) / 30),
        )
       

    def fourier_heart_rate(self, signal):
        # Preprocessing of data using bandpass filter and timeseries PCA

        
        num_intervals = int((len(signal) - self.intervall_length) / 30)
        fourier_rate = np.zeros(num_intervals)
        for i in range(num_intervals):
            # Calculate window signal
            window_signal = signal[30 * i : 30 * i + self.intervall_length]
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
        self.average = (self.POS_kalman + self.CHROM_kalman) / 2

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
        ax.plot(self.ECG_timestamps, self.ECG_rate, label="ECG", color=colors[0])
        ax.plot(self.timestamps, self.POS_kalman_peaks, label="Peak POS", color=colors[1])
        ax.plot(self.timestamps, self.CHROM_kalman_peaks, label="Peak CHROM", color=colors[2])
        self.set_hr_plot_params(ax)

    def plot_fourier_hr(self, ax, colors):
        """
        Plot the Fourier estimated HR from POS and CHROM signals and their average against the ECG data on the given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
            colors (list): List of colors for the plot.
        """
        ax.plot(self.ECG_timestamps, self.ECG_rate, label="ECG", color=colors[0])
        ax.plot(self.timestamps, self.POS_kalman, label="Fourier POS", color=colors[1])
        ax.plot(self.timestamps, self.CHROM_kalman, label="Fourier CHROM", color=colors[2])
        ax.plot(self.timestamps, self.average, label="POS CHROM Average", color=colors[3])
        self.set_hr_plot_params(ax)

            
    def set_hr_plot_params(self, ax):
        ax.set_ylabel("Heart rate (bpm)", fontsize=20)
        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_ylim(100, 200)
        ax.set_title("Peak counting", fontsize=20)
        ax.legend(loc="lower center", fontsize=14, frameon=False, ncol=3)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        

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
            analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
            analysis.vital_sign = vital_sign
            analysis.subject_folder = self.subject_folder
            analysis.ground_truth_signal = self.ECG_rate
            analysis.ground_truth_timestamps = self.ECG_timestamps
            analysis.reference_signal = reference_signal
            analysis.reference_timestamps = self.timestamps
            analysis.run()
        
        # Run the statistical analyses
        run_analysis("POS_Peak_counting", self.POS_kalman_peaks)
        run_analysis("CHROM_Peak_counting", self.CHROM_kalman_peaks)
        run_analysis("POS_Fourier_analysis", self.POS_kalman)
        run_analysis("CHROM_Fourier_analysis", self.CHROM_kalman)
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
            "ECG rate": self.ECG_rate,                   # ECG rate data
            "POS kalman": self.POS_kalman,               # Processed POS signal data
            "CHROM kalman": self.CHROM_kalman,           # Processed CHROM signal data
            "POS kalman peaks": self.POS_kalman_peaks,   # Detected peaks in the POS signal
            "CHROM kalman peaks": self.CHROM_kalman_peaks, # Detected peaks in the CHROM signal
            "Timestamps": self.timestamps,               # General timestamps for the data
            "Average": self.average                      # Average of the POS and CHROM signals
        }
        
        return data

