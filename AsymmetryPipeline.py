import matplotlib.pyplot as plt
import numpy as np
import os
import MeerkatPipelineHelperfunctions
import pandas as pd
import seaborn as sns

class BreathingAsymmetry:
    def __init__(self, data_analysis_folder="", intervall_length=1800):
        """
        Initialize the BreathingAsymmetry class.
        
        Parameters:
        - data_analysis_folder (str): Path to the folder containing data.
        - intervall_length (int): Length of the sliding window interval in seconds.
        """
        self.data_analysis_folder = data_analysis_folder
        self.intervall_length = intervall_length

    def run(self):
        """
        Run the complete analysis process.
        """
        # Choose the subject folder using the helper function
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(self.data_analysis_folder)
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
        self.ts1 = df["Time (s)"].values
        self.ROI_x_1_array = df[" Rectangle x1"].values
        self.ROI_x_2_array = df[" Rectangle x2"].values
        self.ROI_y_1_array = df[" Rectangle y1"].values
        self.ROI_y_2_array = df[" Rectangle y2"].values
        self.left_chest_depth = df[" Depth Left Chest"].values
        self.right_chest_depth = df[" Depth Right Chest"].values

    def calculate_pca_signals(self):
        """
        Calculate PCA signals for the left and right chest respiratory data.
        """
        # Calculate PCA signals for left chest
        self.PCA_signal_left = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.left_chest_depth, self.ROI_x_1_array, self.ROI_x_2_array,
            self.ROI_y_1_array, self.ROI_y_2_array, outliers=False)
        # Calculate PCA signals for right chest
        self.PCA_signal_right = MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            self.right_chest_depth, self.ROI_x_1_array, self.ROI_x_2_array,
            self.ROI_y_1_array, self.ROI_y_2_array, outliers=False)

    def find_tidal_volumes(self):
        """
        Find valid tidal volumes and peaks for the left and right sides.
        """
        # Find tidal volumes and peaks for left chest
        self.left_peaks, self.left_volumes, _, _ = MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
            self.ts1, self.PCA_signal_left)
        # Find tidal volumes and peaks for right chest
        self.right_peaks, self.right_volumes, _, _ = MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(
            self.ts1, self.PCA_signal_right)

    def calculate_windowed_volumes(self):
        """
        Calculate average volumes in sliding windows for both left and right sides.
        """
        # Calculate volumes using sliding windows for left chest
        self.left_volumes_windowed = self.chest_half_movement_sliding_window(
            self.PCA_signal_left, self.left_peaks, self.left_volumes)
        # Calculate volumes using sliding windows for right chest
        self.right_volumes_windowed = self.chest_half_movement_sliding_window(
            self.PCA_signal_right, self.right_peaks, self.right_volumes)

    def chest_half_movement_sliding_window(self, signal, valid_peaks, valid_tidal_volumes):
        """
        Calculate average tidal volumes in sliding windows.
        
        Parameters:
        - signal (array): PCA signal array.
        - valid_peaks (array): Array of valid peak indices.
        - valid_tidal_volumes (array): Array of valid tidal volumes.
        
        Returns:
        - interval_volumes (list): Average tidal volumes in sliding windows.
        """
        interval_volumes = []
        valid_peaks = np.array(valid_peaks)

        # Iterate over the signal with a sliding window approach
        for i in range(0, len(signal) - self.intervall_length, 30):
            indices = np.where((valid_peaks >= i) & (valid_peaks <= self.intervall_length + i))[0]
            if indices.size > 0:
                avg_volume = np.mean(valid_tidal_volumes[indices])
            else:
                avg_volume = interval_volumes[-1] if interval_volumes else 0.1
            interval_volumes.append(avg_volume)

        return interval_volumes

    def generate_timestamps(self):
        """
        Generate timestamps for the sliding windows.
        """
        # Create an array of timestamps for the sliding windows
        self.window_timestamps = np.linspace(
            self.ts1[self.intervall_length], self.ts1[-1],
            num=len(self.left_volumes_windowed))

    def plot_data(self):
        """
        Plot the asymmetry score over time.
        """
        # Set plotting style and parameters
        colors=MeerkatPipelineHelperfunctions.set_plot_params()

        # Calculate the asymmetry score
        self.asymmetry_score = 200 * (np.array(self.left_volumes_windowed) - np.array(self.right_volumes_windowed)) / \
                               (np.array(self.left_volumes_windowed) + np.array(self.right_volumes_windowed))

        # Create a plot
        fig, ax = plt.subplots(figsize=(7.4, 4))
        ax.plot(self.window_timestamps, self.asymmetry_score, color=colors[0])
        ax.set(xlabel="Time (s)", ylabel="Asymmetry score (%)", title="Breathing asymmetry")
        # Customize plot appearance
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def return_data(self):
        """
        Return the calculated timestamps and asymmetry scores.
        
        Returns:
        - window_timestamps (array): Timestamps for sliding windows.
        - asymmetry_score (array): Calculated breathing asymmetry scores.
        """
        return self.window_timestamps, self.asymmetry_score
