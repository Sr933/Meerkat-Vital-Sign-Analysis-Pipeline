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
import seaborn as sns


class Calculate_heart_rate:
    def __init__(self):
        self.intervall_length = 3600
        self.data_analysis_folder = ""

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
        self.POS()

        # Calculate heart rate from POS and CHROM signals using peak counting
        self.timestamps, self.heart_rate_CHROM = self.find_heart_rate(
            self.heart_rate_signal_CHROM
        )
        _, self.heart_rate_POS = self.find_heart_rate(
            self.heart_rate_signal_POS_filtered
        )
        # Calculate heart from POS and CHROM signals using Fourier analysis with Bayesian Inference
        CHROM_fourier = self.fourier_heart_rate(self.heart_rate_signal_CHROM)
        POS_fourier = self.fourier_heart_rate(self.heart_rate_signal_POS)

        # Kalman filter the obtained heart rates
        self.CHROM_kalman_peaks = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.heart_rate_CHROM, 0.0001, 0.00001, 20
        )
        self.POS_kalman_peaks = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.heart_rate_POS, 0.0001, 0.00001, 20
        )
        self.CHROM_kalman = MeerkatPipelineHelperfunctions.Kalman_filter(
            CHROM_fourier, 0.0001, 0.00001, 20
        )
        self.POS_kalman = MeerkatPipelineHelperfunctions.Kalman_filter(
            POS_fourier, 0.0001, 0.00001, 20
        )
        self.ECG_rate = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.ECG_rate, 0.0001, 0.00001, 20
        )

    def import_ECG_rate(self):
        # Define ECG file folder
        hr_folder = repr("\\" + "ECG heart rate")
        hr_folder = self.subject_folder + hr_folder[2:-1]
        if len(os.listdir(hr_folder)) > 0:  # if file exists import data from it
            hr_file = repr("\\" + os.listdir(hr_folder)[0])
            hr_filepath = hr_folder + hr_file[2:-1]
            # Load data from csv
            df = pd.read_csv(hr_filepath)
            self.ECG_timestamps = np.array(df["Time (s)"])
            self.ECG_rate = np.array(df[" Heart rate (bpm)"])
        else:
            print("No valid heart rate data")
            sys.exit()

    def import_heart_camera_data(self):
        # Define camera filepath
        camera_folder = repr("\\" + "RGB-D camera video data")
        camera_folder = self.subject_folder + camera_folder[2:-1]
        camera_file = repr("\\" + os.listdir(camera_folder)[0])
        camera_filepath = camera_folder + camera_file[2:-1]
        # Load data from csv file
        df = pd.read_csv(camera_filepath)
        self.ts1 = np.array(df["Time (s)"])
        self.red_signal = np.array(df[" Red"])
        self.green_signal = np.array(df[" Green"])
        self.blue_signal = np.array(df[" Blue"])
        self.mean_red = np.mean(self.red_signal)
        self.mean_blue = np.mean(self.blue_signal)
        self.mean_green = np.mean(self.green_signal)

    def CHROM_signal(self):
        # Algortithm after https://ieeexplore.ieee.org/document/6523142

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

    def POS(self):
        # Algorithm after https://ieeexplore.ieee.org/document/7565547

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
            self.butter, self.heart_rate_signal_POS
        )

    def find_heart_rate(self, heart_rate_signal):
        # Find peaks in signal
        peak_location = []
        peaks, _ = find_peaks(heart_rate_signal, distance=6, height=0)
        for i in range(len(peaks) - 1):
            peak_location.append(heart_rate_signal[peaks[i]])

        intervall_peak_numbers = []

        # Count number of peaks in intervall iterating over signal length
        for i in range(int((len((heart_rate_signal)) - self.intervall_length) / 30)):
            result = np.where(
                np.logical_and(peaks >= 30 * i, peaks <= self.intervall_length + 30 * i)
            )[0]
            if len(result) > 0:
                first_peak = peaks[result[0]]
                last_peak = peaks[result[-1]]
                intervall_peak_numbers.append(
                    (len(result) - 1) * 60 / (last_peak - first_peak) * 30
                )
            else:
                intervall_peak_numbers.append(0)
        # Calculate timestamps of intervalls
        time_intervall_peak_numbers = np.linspace(
            self.ts1[self.intervall_length],
            self.ts1[-1],
            num=int((len((heart_rate_signal)) - self.intervall_length) / 30),
        )
        return time_intervall_peak_numbers, intervall_peak_numbers

    def fourier_heart_rate(self, signal):
        # Preprocessing of data using bandpass filter and timeseries PCA

        fourier_rate = []
        for i in range(int(((len(signal)) - self.intervall_length) / 30)):
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
            yf = np.multiply(yf, resp_gaussian) * 100
            fourier_rate.append(xf[np.where(yf == np.max(yf))[0][0]])
        return fourier_rate

    def plot_data(self):
        # Plot all obtained data in comparison to the ground truth ECG

        # Calculate average of CHROM and POS
        self.average = np.zeros(len(self.timestamps))
        for i in range(len(self.timestamps)):
            self.average[i] = (self.POS_kalman[i] + self.CHROM_kalman[i]) / 2

        # Plot parameters
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        colors = sns.color_palette("deep")
        ax1.plot(self.ECG_timestamps, self.ECG_rate, label="ECG", color=colors[0])
        ax1.plot(
            self.timestamps, self.POS_kalman_peaks, label="Peak POS", color=colors[1]
        )
        ax1.plot(
            self.timestamps,
            self.CHROM_kalman_peaks,
            label="Peak CHROM",
            color=colors[2],
        )
        ax1.set_ylabel("Heart rate (bpm)", fontsize=20)
        ax1.set_xlabel("Time (s)", fontsize=20)
        ax1.set_ylim(100, 200)
        ax1.set_title("Peak counting", fontsize=20)
        ax1.legend(loc="lower center", fontsize=14, frameon=False, ncol=3)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position("left")
        ax1.xaxis.set_ticks_position("bottom")
        ax1.tick_params(axis="x", labelsize=14)
        ax1.tick_params(axis="y", labelsize=14)

        ax2.plot(self.ECG_timestamps, self.ECG_rate, label="ECG", color=colors[0])
        ax2.plot(self.timestamps, self.POS_kalman, label="Fourier POS", color=colors[1])
        ax2.plot(
            self.timestamps, self.CHROM_kalman, label="Fourier CHROM", color=colors[2]
        )
        ax2.plot(
            self.timestamps, self.average, label="POS CHROM Average", color=colors[3]
        )
        ax2.set_ylabel("Heart rate (bpm)", fontsize=20)
        ax2.set_xlabel("Time (s)", fontsize=20)
        ax2.set_ylim(100, 200)
        ax2.set_title("Fourier analysis", fontsize=20)
        ax2.legend(loc="lower center", fontsize=14, frameon=False, ncol=2)
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
        # Perform statistical analysis of each of the obtained signals and write intermediate files
        print("POS Peak counting")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "POS_Peak_counting"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ECG_rate
        Analysis.ground_truth_timestamps = self.ECG_timestamps
        Analysis.reference_signal = self.POS_kalman_peaks
        Analysis.reference_timestamps = self.timestamps
        Analysis.kappa = 10
        Analysis.run()
        print("CHROM peak counting")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "CHROM_Peak_counting"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ECG_rate
        Analysis.ground_truth_timestamps = self.ECG_timestamps
        Analysis.reference_signal = self.CHROM_kalman_peaks
        Analysis.reference_timestamps = self.timestamps
        Analysis.run()
        print("POS Fourier analysis")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "POS_Fourier_analysis"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ECG_rate
        Analysis.ground_truth_timestamps = self.ECG_timestamps
        Analysis.reference_signal = self.POS_kalman
        Analysis.reference_timestamps = self.timestamps
        Analysis.run()
        print("CHROM Fourier analysis")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "CHROM_Fourier_analysis"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ECG_rate
        Analysis.ground_truth_timestamps = self.ECG_timestamps
        Analysis.reference_signal = self.CHROM_kalman
        Analysis.reference_timestamps = self.timestamps
        Analysis.run()
        print("CHROM and POS")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "CHROM_POS"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ECG_rate
        Analysis.ground_truth_timestamps = self.ECG_timestamps
        Analysis.reference_signal = self.average
        Analysis.reference_timestamps = self.timestamps
        Analysis.run()

    def return_data(self):
        return (
            self.ECG_timestamps,
            self.ECG_rate,
            self.POS_kalman,
            self.CHROM_kalman,
            self.POS_kalman_peaks,
            self.CHROM_kalman_peaks,
            self.timestamps,
            self.average,
        )
