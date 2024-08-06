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
    def __init__(self):
        self.data_analysis_folder = ""
        self.time_shift = 0
        self.intervall_length_resp = 1800
        self.intervall_length_vol = 1800

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
        left_chest_depth_PCA = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.left_chest_depth,
            self.ROI_x_1_array,
            self.ROI_x_2_array,
            self.ROI_y_1_array,
            self.ROI_y_2_array,
        )
        right_chest_depth_PCA = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.right_chest_depth,
            self.ROI_x_1_array,
            self.ROI_x_2_array,
            self.ROI_y_1_array,
            self.ROI_y_2_array,
        )
        left_abdomen_depth_PCA = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.left_abdomen_depth,
            self.ROI_x_1_array,
            self.ROI_x_2_array,
            self.ROI_y_1_array,
            self.ROI_y_2_array,
        )
        right_abdomen_depth_PCA = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.right_abdomen_depth,
            self.ROI_x_1_array,
            self.ROI_x_2_array,
            self.ROI_y_1_array,
            self.ROI_y_2_array,
        )

        # Combine signals for each quadrant, very abnormal values were changed to 10000 to easily remove them
        signal_sum = np.zeros(len(left_abdomen_depth_PCA))
        signal_number = np.zeros(len(left_abdomen_depth_PCA))
        for i in range(len(left_abdomen_depth_PCA)):
            if right_abdomen_depth_PCA[i] != 10000 and self.right_abdomen_depth[i] > 0:
                signal_sum[i] += right_abdomen_depth_PCA[i]
                signal_number[i] += 1
            if left_abdomen_depth_PCA[i] != 10000 and self.left_abdomen_depth[i] > 0:
                signal_sum[i] += left_abdomen_depth_PCA[i]
                signal_number[i] += 1
            if right_chest_depth_PCA[i] != 10000 and self.right_chest_depth[i] > 0:
                signal_sum[i] += right_chest_depth_PCA[i]
                signal_number[i] += 1
            if left_chest_depth_PCA[i] != 10000 and self.left_chest_depth[i] > 0:
                signal_sum[i] += left_chest_depth_PCA[i]
                signal_number[i] += 1
            if signal_number[i] == 0:
                signal_number[i] = 1
                signal_sum[i] = 0
        self.PCA_signal_vol = np.divide(signal_sum, signal_number)

        # Calculate resp rate using peak counting and tidal volume
        self.valid_peaks, self.valid_tidal_volumes, _, _ = (
            MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
                self.ts1, self.PCA_signal_vol
            )
        )
        # Bin the measured peaks and volumes into intervals
        self.find_volume_and_rate_in_sliding_window()
        self.timestamps_intervalls_rate = np.linspace(
            self.ts1[self.intervall_length_resp],
            self.ts1[-1],
            num=int((len((self.PCA_signal_vol)) - self.intervall_length_resp) / 30),
        )
        # Calculate resp rate using Fourier analysis of the signal
        self.respiratory_rate_fourier()

        # Estimate the true state of the system with Kalman filters of the measurements
        self.kalman_breathing_rate = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.fourier_breathing_rate, 0.0001, 0.00001, 20
        )
        self.kalman_breathing_rate_peaks = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.intervall_resp_rate, 0.0001, 0.00001, 20
        )

        self.intervall_volumes = MeerkatPipelineHelperfunctions.Kalman_filter(
            self.intervall_volumes, 0.00001, 0.000001, 4
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
                    self.ventilator_rate_windowed_values, 0.0001, 0.00001, 20
                )
            )

            # Use the same Kalman filter as on the camera data for comparative results
            self.ventilator_volume_windowed_values = (
                MeerkatPipelineHelperfunctions.Kalman_filter(
                    self.ventilator_volume_windowed_values, 0.00001, 0.000001, 4
                )
            )
            self.v_in_w = MeerkatPipelineHelperfunctions.Kalman_filter(
                self.v_in_w, 0.00001, 0.000001, 4
            )
            self.v_ex_w = MeerkatPipelineHelperfunctions.Kalman_filter(
                self.v_ex_w, 0.00001, 0.000001, 4
            )

    def import_camera_resp_data(self):
        # Define camera folder
        camera_folder = repr("\\" + "RGB-D camera video data")
        camera_folder = self.subject_folder + camera_folder[2:-1]
        camera_file = os.listdir(camera_folder)[0]
        camera_file = repr("\\" + camera_file)
        camera_filepath = camera_folder + camera_file[2:-1]

        # Extracr parameters from csv file
        df = pd.read_csv(camera_filepath)
        self.ts1 = np.array(df["Time (s)"])
        self.average_depth = np.array(df[" Depth"])
        self.ROI_x_1_array = np.array(df[" Rectangle x1"])
        self.ROI_x_2_array = np.array(df[" Rectangle x2"])
        self.ROI_y_1_array = np.array(df[" Rectangle y1"])
        self.ROI_y_2_array = np.array(df[" Rectangle y2"])
        self.left_chest_depth = np.array(df[" Depth Left Chest"])
        self.right_chest_depth = np.array(df[" Depth Right Chest"])
        self.left_abdomen_depth = np.array(df[" Depth Left Abdomen"])
        self.right_abdomen_depth = np.array(df[" Depth Right Abdomen"])

    def import_ventilator_rate_and_volume(self):
        # Import rate
        rate_folder = repr("\\" + "Ventilator respiratory rate")
        rate_folder = self.subject_folder + rate_folder[2:-1]
        self.has_ventilator = False

        # Read ventilator data if available in database
        if len(os.listdir(rate_folder)) > 0:
            self.has_ventilator = True

            # Import ventilator rate from csv
            rate_file = os.listdir(rate_folder)[0]
            rate_file = repr("\\" + rate_file)
            rate_filepath = rate_folder + rate_file[2:-1]
            df = pd.read_csv(rate_filepath)
            self.timestamps_ventilator_rate = np.array(df["Time (s)"])
            self.ventilator_rate = np.array(df[" Respiratory rate (breaths/min)"])

            # Import tidal volume best estimates from ventilator from csv
            volume_folder = repr("\\" + "Ventilator tidal volume best estimate")
            volume_folder = self.subject_folder + volume_folder[2:-1]
            camera_file = os.listdir(volume_folder)[0]
            camera_file = repr("\\" + camera_file)
            camera_filepath = volume_folder + camera_file[2:-1]
            df2 = pd.read_csv(camera_filepath)
            self.timestamps_ventilator_volume_best_estimate = np.array(df2["Time (s)"])
            self.ventilator_tidal_volume_best_estimate = np.array(df2[" VT (ml)"])

            # Import upper lower estimation for tidal volume from measurements of inhalation and exhalation from csv
            volume_folder = repr("\\" + "Ventilator tidal volume upper lower")
            volume_folder = self.subject_folder + volume_folder[2:-1]
            camera_file = os.listdir(volume_folder)[0]
            camera_file = repr("\\" + camera_file)
            camera_filepath = volume_folder + camera_file[2:-1]
            df3 = pd.read_csv(camera_filepath)
            self.timestamps_ventilator_volume_upper_lower = np.array(df2["Time (s)"])
            self.ventilator_volume_exhalation = np.array(df3[" VT_Expiration (ml)"])
            self.ventilator_volume_inhalation = np.array(df3[" VT_Inspiration (ml)"])

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
        volume_i = 0
        valid_peaks = np.array(self.valid_peaks)

        # Iterate over signal for each window to find number of peaks and corresponding tidal volumes
        for i in range(
            int((len((self.PCA_signal_vol)) - self.intervall_length_vol) / 30)
        ):
            # Find peaks in intervall
            result = np.where(
                np.logical_and(
                    valid_peaks >= 30 * i,
                    valid_peaks <= self.intervall_length_vol + 30 * i,
                )
            )[0]
            # If peaks found calculate respiratory rate and volumes from them
            if len(result) > 1:
                first_peak = valid_peaks[result[0]]
                last_peak = valid_peaks[result[-1]]
                self.intervall_resp_rate.append(
                    (len(result) - 1) * 60 / (last_peak - first_peak) * 30
                )
                volume_i = []
                for j in range(len(result)):
                    volume_i.append(self.valid_tidal_volumes[result[j]])
                self.intervall_volumes.append(np.mean(volume_i))
            # If no peaks found, asign previous resp rate and tidal volume, if no measurements have been made, assign 0
            else:
                if len(self.intervall_volumes) > 0:
                    self.intervall_volumes.append(self.intervall_volumes[-1])
                else:
                    self.intervall_volumes.append(5)
                if len(self.intervall_resp_rate) > 0:
                    self.intervall_resp_rate.append(self.intervall_resp_rate[-1])
                else:
                    self.intervall_resp_rate.append(0)
        # Calculate timestamps for the intervals
        self.timestamps_intervalls_vol = np.linspace(
            self.ts1[self.intervall_length_vol],
            self.ts1[-1],
            num=int((len((self.PCA_signal_vol)) - self.intervall_length_vol) / 30),
        )

    def respiratory_rate_fourier(self):
        self.fourier_breathing_rate = []
        peaks_in_bins = []
        for i in range(
            int(((len(self.PCA_signal_resp)) - self.intervall_length_resp) / 30)
        ):

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
            yf = np.multiply(yf, resp_gaussian) * 100
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
        #Plot parameters
        plt.style.use(['default'])
        params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : False,
          "font.family" : "serif",
          "font.sans-serif": "Helvetica",
          }
        plt.rcParams.update(params)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Plot ventilator tidal volume best estimates and upper lower bounds if available
        plt.style.use(['default'])
        colors = sns.color_palette("deep")
        if self.has_ventilator:
            ax1.plot(
                np.array(self.timestamps_ventilator_volume_windowed_values)/60,
                self.ventilator_volume_windowed_values,
                
                label="Ventilator",color=colors[0]
            )

            ax1.fill_between(
                np.array(self.time_in_w)/60,
                self.v_in_w,
                self.v_ex_w,
                color=colors[0],
                alpha=0.4,
                label="Ventilator bounds",
            )
        # Plot camera estimates
        ax1.plot(
            np.array(self.timestamps_intervalls_vol)/60,
            self.intervall_volumes,
            label="Camera",color=colors[1]
        )
        

        ax1.set_xlabel("Time (min)", fontsize=20)
        ax1.set_ylabel("Tidal volume (ml)", fontsize=20)
        ax1.set_ylim(2, 9)
        ax1.set_title("Tidal volume", fontsize=20)
        ax1.yaxis.set_ticks_position("left")
        ax1.xaxis.set_ticks_position("bottom")
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.legend(loc='upper right', frameon=False, fontsize=14)
        
        print("Mean tidal volume(ml): ", np.mean(self.intervall_volumes))

        # Plot ventilator rate if available
        if self.has_ventilator:
            ax2.plot(
                np.array(self.timestamps_ventilator_volume_windowed_values)/60,
                self.ventilator_rate_windowed_values,
                color=colors[0],
                label="Ventilator",
            )
        # Plot camera respiratory rate
        ax2.plot(
            np.array(self.timestamps_intervalls_rate)/60, self.kalman_breathing_rate, label="Camera", color=colors[1]
        )
        #ax2.plot(
        #    np.array(self.timestamps_intervalls_rate)/60, self.kalman_breathing_rate_peaks, label="Peaks", color=colors[2]
        #)
        ax2.set_xlabel("Time (min)", fontsize=20)
        ax2.set_ylabel("Respiratory rate (breaths/min)", fontsize=20)
        ax2.set_title("Respiratory rate", fontsize=20)
        ax2.legend(loc="upper left", fontsize=14)
        ax2.set_ylim(15, 80)
        ax2.yaxis.set_ticks_position("left")
        ax2.xaxis.set_ticks_position("bottom")
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.legend(loc='upper right', frameon=False, fontsize=14)


        plt.tight_layout(pad=2.5,w_pad=2.5)
        plt.show()

    def statistical_analysis(self):
        # Perform statistical analysis for the different measurement and estimation techniques
        print("Fourier analysis")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "Resp_Fourier"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ventilator_rate_windowed_values
        Analysis.ground_truth_timestamps = (
            self.timestamps_ventilator_volume_windowed_values
        )
        Analysis.reference_signal = self.kalman_breathing_rate
        Analysis.reference_timestamps = self.timestamps_intervalls_rate
        Analysis.run()

        print("Resp Peak counting")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "Resp_Peak_counting"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ventilator_rate_windowed_values
        Analysis.ground_truth_timestamps = (
            self.timestamps_ventilator_volume_windowed_values
        )
        Analysis.reference_signal = self.kalman_breathing_rate_peaks
        Analysis.reference_timestamps = self.timestamps_intervalls_vol
        Analysis.run()

        print("Tidal volume best estimate")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "Tidal_volume"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_signal = self.ventilator_volume_windowed_values
        Analysis.ground_truth_timestamps = (
            self.timestamps_ventilator_volume_windowed_values
        )
        Analysis.reference_signal = self.intervall_volumes
        Analysis.reference_timestamps = self.timestamps_intervalls_vol
        Analysis.run()

        print("Tidal volume upper lower")
        Analysis = StatisticalAnalysis.MeerkatStatisticalAnalysis()
        Analysis.vital_sign = "Tidal_volume_upper_lower"
        Analysis.subject_folder = self.subject_folder
        Analysis.ground_truth_timestamps = self.time_in_w
        Analysis.tidalvolumeupper = self.v_ex_w
        Analysis.tidalvolumelower = self.v_in_w
        Analysis.reference_signal = self.intervall_volumes
        Analysis.reference_timestamps = self.timestamps_intervalls_vol
        Analysis.tidalvolumeflag = True
        Analysis.run()
    def return_data(self):
        return self.timestamps_ventilator_volume_windowed_values,self.ventilator_volume_windowed_values, self.ventilator_rate_windowed_values, self.time_in_w, self.v_in_w, self.v_ex_w,self.timestamps_intervalls_vol, self.intervall_volumes,self.timestamps_intervalls_rate, self.kalman_breathing_rate, self.kalman_breathing_rate_peaks