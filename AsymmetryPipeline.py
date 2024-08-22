import matplotlib.pyplot as plt
import numpy as np
import os
import MeerkatPipelineHelperfunctions
import pandas as pd


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

        # Create an array of start indices for each interval
        start_indices = np.arange(0, len(signal) - self.intervall_length, 30)

        # Calculate the end indices for each interval
        end_indices = start_indices + self.intervall_length

        # Create a boolean mask where each element is True if the corresponding peak is within the interval
        mask = (valid_peaks[:, np.newaxis] >= start_indices) & (
            valid_peaks[:, np.newaxis] <= end_indices
        )

        # Use the mask to find the indices of the peaks within each interval
        indices = np.where(mask)

        # Calculate the average tidal volume for each interval
        interval_volumes = np.array(
            [
                (
                    np.mean(valid_tidal_volumes[indices[0][indices[1] == i]])
                    if np.any(indices[1] == i)
                    else (interval_volumes[-1] if interval_volumes else 0.1)
                )
                for i in range(len(start_indices))
            ]
        )

        return interval_volumes

    def generate_timestamps(self):
        """
        Generate timestamps for the sliding windows.
        """
        # Create an array of timestamps for the sliding windows
        self.window_timestamps = np.linspace(
            self.ts1[self.intervall_length],
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
