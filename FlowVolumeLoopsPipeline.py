import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import csv
from scipy.signal import sosfiltfilt
import MeerkatPipelineHelperfunctions


class CalculateFlowVolumeLoop:
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
                self.ts1, self.PCA_signal_vol
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
        for j in range(len(self.valid_peaks) - 1):
            breath_volume, breath_flow, flow_crossings = self.camera_loop_calc(j)
            # Criteria for fv loop visualisation
            if self.valid_loop(breath_flow, breath_volume, flow_crossings):
                ax1.plot(breath_volume, breath_flow, color=colors[j % 10])
                self.camera_loops_volume.append(breath_volume)
                self.camera_loops_flow.append(breath_flow)

        # Define plot parameters
        self.set_fv_plot_params(ax1)

        # Calculate ventilator loop
        if self.hasventilator:
            # Iterate over peaks and calculate flow volume loop
            for j in range(len(self.breath_start) - 1):
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
        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def plot_single_loops(self):
        # Plot parameters
        colors = MeerkatPipelineHelperfunctions.set_plot_params()
        
        for i in range(len(self.valid_peaks)):
            camera_peak_t = self.ts1[i]
            for j in range(len(self.breath_start)):
                ventilator_peak_t = self.ventilator_time[int(self.breath_start[j])]
                if abs(camera_peak_t - ventilator_peak_t) < 2.0:
                    plot1 = False
                    plot2 = False
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # Define plot parameters
                    self.set_fv_plot_params(ax1)

                    breath_volume, breath_flow, flow_crossings = self.camera_loop_calc(j)
                    if self.valid_loop(breath_flow, breath_volume, flow_crossings):
                        ax1.plot(breath_volume, breath_flow, color=colors[0])
                        plot1 = True

                        breath_volume, breath_flow, flow_crossings = self.camera_loop_calc(j)

                        if self.valid_loop(breath_flow, breath_volume, flow_crossings):
                            ax2.plot(breath_volume, breath_flow, color=colors[0])
                            plot2 = True
                    # Define plot parameters
                    self.set_fv_plot_params(ax2)
                    

                    # Show loops if both loops are considered valid
                    if plot1 and plot2:
                        plt.tight_layout(pad=2.5, w_pad=2.5)
                        plt.show()
                    else:
                        plt.clf()
                        plt.close()

    def set_fv_plot_params(self, ax):
        ax.set_xlabel("Volume (ml)", fontsize=20)
        ax.set_ylabel("Flow (ml/s)", fontsize=20)
        ax.set_title("Ventilator", fontsize=20)
        ax.set_xlim(-10, 1)
        ax.set_ylim(-60, 60)
        ax.grid()
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

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

