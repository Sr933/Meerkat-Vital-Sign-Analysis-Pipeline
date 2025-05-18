from MeerkatPipeline import (
    HeartRatePipeline,
    PulseOxygenationPipeline,
    FlowVolumeLoopPipeline,
    RespiratoryPipeline,
    BreathingAsymmetryPipeline,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import SummaryStatistics
import BA_plot
import pandas as pd

database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"

ox = PulseOxygenationPipeline(database_path, 900)
ox.run()
# 5 for image
data=ox.return_data()
(pulse_oximeter_timestamps, pulse_oximeter_spO2, time_spO2, spO2_calibrated, spO2_rgb, spO2_infrared, spO2_ycgcr, timestamps_spO2_calibrated)=(
    data["Pulse Oximeter Timestamps"],
            data["Pulse Oximeter SpO2"],
            data["Time SpO2"],
            data["SpO2 Calibrated"],
            data["SpO2 RGB"],
            data["SpO2 Infrared"],
            data["SpO2 YCgCr"],
            data["Timestamps SpO2 Calibrated"],)



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
fig, axs = plt.subplots(2, 3, figsize=(8.27, 4.5))

colors = sns.color_palette("deep")

axs[0,0].plot(
            pulse_oximeter_timestamps,
            pulse_oximeter_spO2,
            label="Pulse oximeter",
            color=colors[0],
        )
axs[0,0].plot(time_spO2, spO2_infrared, label="Infrared", color=colors[1])
axs[0,0].plot(time_spO2, spO2_rgb, label="RGB", color=colors[2])
axs[0,0].set_ylabel("SpO2 (%)", fontsize=8)
axs[0,0].set_xlabel("Time (s)", fontsize=8)
axs[0,0].set_title("(a) Infrared and RGB", fontsize=8)
axs[0,0].set_ylim(68, 105)
axs[0,0].legend(loc="lower right", fontsize=8, frameon=False)
axs[0,0].spines["right"].set_visible(False)
axs[0,0].spines["left"].set_visible(False)
axs[0,0].spines["top"].set_visible(False)
axs[0,0].yaxis.set_ticks_position("left")
axs[0,0].xaxis.set_ticks_position("bottom")
axs[0,0].tick_params(axis="x", labelsize=8)
axs[0,0].tick_params(axis="y", labelsize=8)

axs[1,0].plot(
    pulse_oximeter_timestamps,
    pulse_oximeter_spO2,
    label="Pulse oximeter",
    color=colors[0],
)
axs[1,0].plot(time_spO2, spO2_ycgcr, label="YCgCr", color=colors[1])
axs[1,0].plot(
    timestamps_spO2_calibrated,
    spO2_calibrated,
    label="Calibration free",
    color=colors[2],
)
axs[1,0].set_ylabel("SpO2 (%)", fontsize=8)
axs[1,0].set_xlabel("Time (s)", fontsize=8)
axs[1,0].set_title("(b) YCgCr and calibration free", fontsize=8)
axs[1,0].set_ylim(68, 105)
axs[1,0].legend(loc="lower right", fontsize=8, frameon=False)
axs[1,0].spines["right"].set_visible(False)
axs[1,0].spines["left"].set_visible(False)
axs[1,0].spines["top"].set_visible(False)
axs[1,0].yaxis.set_ticks_position("left")
axs[1,0].xaxis.set_ticks_position("bottom")
axs[1,0].tick_params(axis="x", labelsize=8)
axs[1,0].tick_params(axis="y", labelsize=8)

vital_sign = "infrared_oxygen_saturation"
data_analysis_folder = database_path
# Get all subjects in dataset
subjects = []
truth = []
camera = []
subject_number = 0
for file in os.listdir(data_analysis_folder):
    if "mk" in file:
        subjects.append(os.fsdecode(file))

# Iterate over subjects to import all data
for subject in subjects:
    # Define subject folder
    file_folder = repr("\\" + subject)
    file_folder = file_folder[2:-1]
    subject_folder = data_analysis_folder + file_folder
    signal_folder = repr("\\" + "Pipeline results")
    signal_folder = subject_folder + signal_folder[2:-1]

    # Try to import file, if not present skip the subject
    try:
        if len(os.listdir(signal_folder)) > 0:
            # Iterate over all files available to find the matching intermediate file
            for file in os.listdir(signal_folder):
                if vital_sign in file:
                    signal_file = os.listdir(signal_folder)[0]
                    signal_file = repr("\\" + vital_sign)
                    signal_filepath = signal_folder + signal_file[2:-1]

                    df=pd.read_csv(signal_filepath)
                    c=df["Reference Signal"].to_numpy()
                    t=df["Ground Truth Signal"].to_numpy()
                    # Append all elements of c to camera and t to truth
                    camera.extend(c)  # Use extend to add all elements
                    truth.extend(t)   # Use extend to add all elements
                    subject_number += 1

    except:
        continue
print("Number of subjects: ", subject_number)
truth = np.array(truth)
camera = np.array(camera)



# BA plot
BA_plot.mean_diff_plot(
    camera, truth, colors[0], ax=axs[0, 1]
)
axs[0, 1].set_ylabel("Difference (%)", fontsize=8)
axs[0, 1].set_xlabel("Means (%)", fontsize=8)
axs[0,1].set_title("(c) $SpO_2$ BA", fontsize=8)
axs[0, 1].spines["right"].set_visible(False)
axs[0, 1].spines["left"].set_visible(False)
axs[0, 1].spines["top"].set_visible(False)
axs[0, 1].yaxis.set_ticks_position("left")
axs[0, 1].xaxis.set_ticks_position("bottom")
axs[0, 1].tick_params(axis="x", labelsize=8)
axs[0, 1].tick_params(axis="y", labelsize=8)


### Difference distribution
difference = camera - truth
bins = (
    range(int(min(difference)), int(max(difference)) + 1, 1)
    if vital_sign != "tidalvolume"
    else np.linspace(int(min(difference)), int(max(difference)), 36)
)
axs[0, 2].hist(difference, bins=bins, color=colors[0])
axs[0, 2].set_xlabel("Camera - Ground Truth (%)", fontsize=8)
axs[0, 2].set_ylabel("Frequency", fontsize=8)
axs[0,2].set_title("(e) $SpO_2$ errors", fontsize=8)
axs[0, 2].spines["right"].set_visible(False)
axs[0, 2].spines["left"].set_visible(False)
axs[0, 2].spines["top"].set_visible(False)
axs[0, 2].yaxis.set_ticks_position("left")
axs[0, 2].xaxis.set_ticks_position("bottom")
axs[0, 2].tick_params(axis="x", labelsize=8)
axs[0, 2].tick_params(axis="y", labelsize=8)


# Correlation plot with linear regression
spacing=60
axs[1, 1].scatter(truth[::spacing], camera[::spacing], color=colors[0], s=2.5)
axs[1, 1].set_ylabel("Camera (%)", fontsize=8)
axs[1, 1].set_xlabel("Ground Truth (%)", fontsize=8)
axs[1,1].set_title("(d) $SpO_2$ correlation", fontsize=8)
axs[1, 1].set_xlim(min(min(truth), min(camera)) - 5, max(max(truth), max(camera)) + 5)
axs[1, 1].set_ylim(min(min(truth), min(camera)) - 5, max(max(truth), max(camera)) + 5)
axs[1, 1].plot([0, 200], [00, 200], "k--", label="ideal fit", linewidth=1)
axs[1, 1].legend(loc="upper left", fontsize=8, frameon=False)

axs[1, 1].spines["right"].set_visible(False)
axs[1, 1].spines["left"].set_visible(False)
axs[1, 1].spines["top"].set_visible(False)
axs[1, 1].yaxis.set_ticks_position("left")
axs[1, 1].xaxis.set_ticks_position("bottom")
axs[1, 1].tick_params(axis="x", labelsize=8)
axs[1, 1].tick_params(axis="y", labelsize=8)

# Calculate the coverage probability as a function of error
if "Tidal" not in vital_sign:  # can use integers for all other measurements
    kappas = [1]
    coverage_probability = []
    threshold_reached = False
    i = 0
    # Iterate until error can explain 95% of the data
    while not threshold_reached:
        cp = SummaryStatistics.coverage_probability_abs(truth, camera, kappas[i]) * 100
        coverage_probability.append(cp)

        if cp > 95:
            threshold_reached = True
        else:  # repeat with larger kappa
            i += 1
            kappas.append(i - 1)
else:  # use float for tidal volume as quite small values expected
    k = 0.1
    kappas = [k]
    coverage_probability = []
    threshold_reached = False
    i = 0
    # Iterate until error can explain 95% of the data
    while not threshold_reached:
        cp = SummaryStatistics.coverage_probability_abs(truth, camera, kappas[i]) * 100
        coverage_probability.append(cp)

        if cp > 95:
            threshold_reached = True
        else:  # repeat with larger kappa
            i += 1
            k += 0.1
            kappas.append(k)

axs[1, 2].plot(kappas, coverage_probability, color=colors[0])
axs[1, 2].set_xlabel("Absolute error (%)", fontsize=8)
axs[1, 2].set_ylabel("Coverage probability (%)", fontsize=8)
axs[1,2].set_title("(f) $SpO_2$  CP", fontsize=8)
axs[1, 2].spines["right"].set_visible(False)
axs[1, 2].spines["left"].set_visible(False)
axs[1, 2].spines["top"].set_visible(False)
axs[1, 2].yaxis.set_ticks_position("left")
axs[1, 2].xaxis.set_ticks_position("bottom")
axs[1, 2].tick_params(axis="x", labelsize=8)
axs[1, 2].tick_params(axis="y", labelsize=8)

fig.align_ylabels()
fig.align_xlabels()
plt.tight_layout(pad=1)
plt.show()
