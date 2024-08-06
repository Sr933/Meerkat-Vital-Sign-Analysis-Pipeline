import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import csv
from scipy.signal import sosfiltfilt
import MeerkatPipelineHelperfunctions


class CalculateFlowVolumeLoop:
    def __init__(self, data_analysis_folder="", hasventilator=False, plot_single_loops_flag=False):
        self.data_analysis_folder = data_analysis_folder
        self.hasventilator = hasventilator
        self.plot_single_loops_flag = plot_single_loops_flag
        self.intermediate_file = None
        self.intermediate_folder = None

    def run_part1(self):
        self.subject_folder = MeerkatPipelineHelperfunctions.choose_subject(self.data_analysis_folder)
        self.preprocess_data_intermediate()
        return (self.intermediate_file, True) if self.hasventilator else (None, False)

    def run_part2(self):
        self.preprocess_data()
        print(self.valid_peaks)
        if self.hasventilator:
            self.import_ventiliser_results()
        if self.plot_single_loops_flag and self.hasventilator:
            self.plot_single_loops()
        else:
            self.plot_data()

    def preprocess_data_intermediate(self):
        self.import_camera_resp_data()
        self.import_ventilator_flow()
        if self.hasventilator:
            butter = MeerkatPipelineHelperfunctions.butter_bandpass(15/60, 150/60, 100, order=7)
            self.ventilator_flow = sosfiltfilt(butter, self.ventilator_flow)
            self.ventilator_pressure = sosfiltfilt(butter, self.ventilator_pressure)
            self.write_flow_pressure_intermediate_file()

    def import_camera_resp_data(self):
        camera_folder = os.path.join(self.subject_folder, "RGB-D camera video data")
        camera_file = os.listdir(camera_folder)[0]
        camera_filepath = os.path.join(camera_folder, camera_file)
        df = pd.read_csv(camera_filepath)
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
        ventilator_flow_folder = os.path.join(self.subject_folder, "Ventilator flow")
        if len(os.listdir(ventilator_flow_folder)) > 0:
            self.hasventilator = True
            ventilator_flow_file = os.listdir(ventilator_flow_folder)[0]
            ventilator_filepath = os.path.join(ventilator_flow_folder, ventilator_flow_file)
            df = pd.read_csv(ventilator_filepath)
            self.ventilator_flow = df[" Flow (l/min)"].to_numpy()
            self.ventilator_time = df["Time (s)"].to_numpy()
            self.ventilator_pressure = df[" Pressure (mbar)"].to_numpy()

    def write_flow_pressure_intermediate_file(self):
        self.intermediate_folder = os.path.join(self.data_analysis_folder, "pipeline intermediate")
        if os.path.exists(self.intermediate_folder):
            shutil.rmtree(self.intermediate_folder)
        os.makedirs(self.intermediate_folder)
        self.intermediate_file = os.path.join(self.intermediate_folder, "pipeline intermediate.csv")
        with open(self.intermediate_file, "w", newline="") as file:
            writer = csv.writer(file)
            for t, p, f in zip(self.ventilator_time, self.ventilator_pressure, self.ventilator_flow):
                writer.writerow([t, p, f])

    def preprocess_data(self):
        signals = {
            "left_chest": self.left_chest_depth,
            "right_chest": self.right_chest_depth,
            "left_abdomen": self.left_abdomen_depth,
            "right_abdomen": self.right_abdomen_depth
        }
        PCA_signals = {key: MeerkatPipelineHelperfunctions.PCA_respiratory_signal(
            depth, self.ROI_x_1_array, self.ROI_x_2_array, self.ROI_y_1_array, self.ROI_y_2_array
        ) for key, depth in signals.items()}
        
        signal_sum = np.zeros_like(PCA_signals["left_abdomen"])
        signal_number = np.zeros_like(PCA_signals["left_abdomen"])

        for key in PCA_signals:
            signal_sum += np.where(PCA_signals[key] != 10000, PCA_signals[key], 0)
            signal_number += (PCA_signals[key] != 10000).astype(int)

        signal_number = np.where(signal_number == 0, 1, signal_number)
        self.PCA_signal_vol = signal_sum / signal_number

        self.calculate_flow()
        self.valid_peaks, self.valid_tidal_volumes, self.mean, self.std = MeerkatPipelineHelperfunctions.find_valid_tidal_volumes(self.ts1, self.PCA_signal_vol)

    def calculate_flow(self):
        self.flow = np.diff(self.PCA_signal_vol[:-1], prepend=0) / np.diff(self.ts1)

    def import_ventiliser_results(self):
        ventiliser_file = os.listdir(self.intermediate_folder)[2]
        ventiliser_filepath = os.path.join(self.intermediate_folder, ventiliser_file)
        with open(ventiliser_filepath) as f:
            lines = f.readlines()[1:]
            self.breath_start, self.breath_end = zip(*[line.split(",")[1:3] for line in lines])

    def plot_data(self):
        #Store loops
        self.camera_loops_flow=[]
        self.camera_loops_volume=[]
        self.ventilator_loops_flow=[]
        self.ventilator_loops_volume=[]
        # Plot flow-volume loops depending on whether ventilator loops are available
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
        if self.hasventilator:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 6))
        colors = sns.color_palette("deep")
        
        # Iterate over peaks and calculate flow-volume loop for each peak
        for j in range(len(self.valid_peaks) - 1):
            breath_volume = [0]
            breath_flow = np.array(
                self.flow[self.valid_peaks[j] : self.valid_peaks[j + 1]]
            )
            breath_volume = (
                np.array(
                    self.PCA_signal_vol[self.valid_peaks[j] : self.valid_peaks[j + 1]]
                )
                - self.PCA_signal_vol[self.valid_peaks[j]]
            )
            flow_crossings = ((breath_flow[:-1] * breath_flow[1:]) < 0).sum()
            # Criteria for fv loop visualisation
            if (
                abs(breath_flow[-1]) < 20
                and abs(breath_flow[0]) < 20
                and abs(breath_volume[-1]) < 1
                and max(np.absolute(breath_volume)) > self.mean - self.std
                and max(np.absolute(breath_volume)) < self.mean + self.std
                and flow_crossings < 3
                and max(np.abs(breath_flow)) < 40
                and min(breath_volume) < -2
            ):
                ax1.plot(breath_volume, breath_flow, color=colors[j % 10])
                self.camera_loops_volume.append(breath_volume)
                self.camera_loops_flow.append(breath_flow)

        # Define plot parameters
        ax1.set_xlabel("Volume (ml)", fontsize=20)
        ax1.set_ylabel("Flow (ml/s)", fontsize=20)
        ax1.set_title("Camera", fontsize=20)
        ax1.set_xlim(-10, 1)
        ax1.set_ylim(-60, 60)
        ax1.grid()
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.yaxis.set_ticks_position("left")
        ax1.xaxis.set_ticks_position("bottom")
        ax1.tick_params(axis="x", labelsize=14)
        ax1.tick_params(axis="y", labelsize=14)

        # Calculate ventilator loop
        if self.hasventilator:
            # Iterate over peaks and calculate flow volume loop
            for j in range(len(self.breath_start) - 1):
                breath_volume = [0]
                breath_flow = (
                    -np.array(
                        self.ventilator_flow[
                            int(self.breath_start[j]) : int(self.breath_end[j])
                        ]
                    )
                    * 1000
                    / 60
                )
                breath_time = np.array(
                    self.ventilator_time[
                        int(self.breath_start[j]) : int(self.breath_end[j])
                    ]
                )
                if breath_time[0] > self.ts1[0] and breath_time[-1] < self.ts1[-1]:
                    flow_crossings = ((breath_flow[:-1] * breath_flow[1:]) < 0).sum()
                    # Calculate volume signal for breath thorugh numerical integration
                    for i in range(len(breath_flow) - 1):
                        time_0 = breath_time[i]
                        time_1 = breath_time[i + 1]
                        flow_0 = breath_flow[i]
                        flow_1 = breath_flow[i + 1]
                        breath_volume.append(
                            0.5 * (time_1 - time_0) * (flow_0 + flow_1)
                            + breath_volume[i]
                        )
                    # Criteria for fv loop visualisation
                    if (
                        abs(breath_flow[-1]) < 20
                        and abs(breath_volume[-1]) < 1
                        and abs(breath_flow[0]) < 20
                        and flow_crossings < 3
                        and abs(min(breath_volume)) > 2
                        and max(np.absolute(breath_volume)) > self.mean - self.std
                        and max(np.absolute(breath_volume)) < self.mean + self.std
                        and max(np.abs(breath_flow)) < 40
                    ):
                        ax2.plot(breath_volume, breath_flow, color=colors[j % 10])
                        self.ventilator_loops_volume.append(breath_volume)
                        self.ventilator_loops_flow.append(breath_flow)

            # Define plot parameters
            ax2.set_xlabel("Volume (ml)", fontsize=20)
            ax2.set_ylabel("Flow (ml/s)", fontsize=20)
            ax2.set_title("Ventilator", fontsize=20)
            ax2.set_xlim(-10, 1)
            ax2.set_ylim(-60, 60)
            ax2.grid()
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.yaxis.set_ticks_position("left")
            ax2.xaxis.set_ticks_position("bottom")
            ax2.tick_params(axis="x", labelsize=14)
            ax2.tick_params(axis="y", labelsize=14)
        plt.tight_layout(pad=2.5, w_pad=2.5)
        plt.show()

    def plot_single_loops(self):
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
        colors = sns.color_palette("deep")
        for i in range(len(self.valid_peaks)):
            camera_peak_t = self.ts1[i]
            for j in range(len(self.breath_start)):
                ventilator_peak_t = self.ventilator_time[int(self.breath_start[j])]
                if abs(camera_peak_t - ventilator_peak_t) < 2.0:

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # Define plot parameters
                    ax1.set_xlabel("Volume (ml)", fontsize=20)
                    ax1.set_ylabel("Flow (ml/s)", fontsize=20)
                    ax1.set_title("Camera", fontsize=20)
                    ax1.set_xlim(-10, 1)
                    ax1.set_ylim(-60, 60)
                    ax1.grid()
                    ax1.spines["right"].set_visible(False)
                    ax1.spines["left"].set_visible(False)
                    ax1.spines["top"].set_visible(False)
                    ax1.yaxis.set_ticks_position("left")
                    ax1.xaxis.set_ticks_position("bottom")
                    ax1.tick_params(axis="x", labelsize=14)
                    ax1.tick_params(axis="y", labelsize=14)

                    plot1 = False
                    plot2 = False

                    breath_volume = [0]
                    breath_flow = np.array(
                        self.flow[self.valid_peaks[i] : self.valid_peaks[i + 1]]
                    )
                    breath_volume = (
                        np.array(
                            self.PCA_signal_vol[
                                self.valid_peaks[i] : self.valid_peaks[i + 1]
                            ]
                        )
                        - self.PCA_signal_vol[self.valid_peaks[i]]
                    )
                    flow_crossings = ((breath_flow[:-1] * breath_flow[1:]) < 0).sum()
                    if (
                        abs(breath_flow[-1]) < 20
                        and abs(breath_flow[0]) < 20
                        and abs(breath_volume[-1]) < 1
                        and max(np.absolute(breath_volume)) > self.mean - self.std
                        and max(np.absolute(breath_volume)) < self.mean + self.std
                        and flow_crossings < 3
                        and max(np.abs(breath_flow)) < 40
                        and min(breath_volume) < -2
                    ):
                        ax1.plot(breath_volume, breath_flow, color=colors[0])
                        plot1 = True

                    breath_volume = [0]
                    breath_flow = (
                        -np.array(
                            self.ventilator_flow[
                                int(self.breath_start[j]) : int(self.breath_end[j])
                            ]
                        )
                        * 1000
                        / 60
                    )
                    breath_time = np.array(
                        self.ventilator_time[
                            int(self.breath_start[j]) : int(self.breath_end[j])
                        ]
                    )
                    if breath_time[0] > self.ts1[0] and breath_time[-1] < self.ts1[-1]:
                        flow_crossings = (
                            (breath_flow[:-1] * breath_flow[1:]) < 0
                        ).sum()
                        for i in range(len(breath_flow) - 1):
                            time_0 = breath_time[i]
                            time_1 = breath_time[i + 1]
                            flow_0 = breath_flow[i]
                            flow_1 = breath_flow[i + 1]
                            breath_volume.append(
                                0.5 * (time_1 - time_0) * (flow_0 + flow_1)
                                + breath_volume[i]
                            )

                        if (
                            abs(breath_flow[-1]) < 20
                            and abs(breath_flow[0]) < 20
                            and abs(breath_volume[-1]) < 1
                            and flow_crossings < 3
                            and max(np.abs(breath_flow)) < 40
                            
                        ):
                            ax2.plot(breath_volume, breath_flow, color=colors[0])
                            plot2 = True
                    # Define plot parameters
                    ax2.set_xlabel("Volume (ml)", fontsize=20)
                    ax2.set_ylabel("Flow (ml/s)", fontsize=20)
                    ax2.set_title("Ventilator", fontsize=20)
                    ax2.set_xlim(-10, 1)
                    ax2.set_ylim(-60, 60)
                    ax2.grid()
                    ax2.spines["right"].set_visible(False)
                    ax2.spines["left"].set_visible(False)
                    ax2.spines["top"].set_visible(False)
                    ax2.yaxis.set_ticks_position("left")
                    ax2.xaxis.set_ticks_position("bottom")
                    ax2.tick_params(axis="x", labelsize=14)
                    ax2.tick_params(axis="y", labelsize=14)
                    plt.tight_layout(pad=2.5, w_pad=2.5)

                    # Shot loops if both loops are considered valid
                    if plot1 == True and plot2 == True:
                        plt.show()
                    else:
                        plt.clf()
                        plt.close()
                        
    def return_data(self):
        return self.ventilator_loops_flow, self.ventilator_loops_volume, self.camera_loops_flow, self.camera_loops_volume
    
 
