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

Resp = RespiratoryPipeline(database_path, 1800, 1800)
Resp.run()


data = Resp.return_data()
(
    timestamps_ventilator_volume_windowed_values,
    ventilator_volume_windowed_values,
    ventilator_rate_windowed_values,
    time_in_w,
    v_in_w,
    v_ex_w,
    timestamps_intervalls_vol,
    intervall_volumes,
    timestamps_intervalls_rate,
    kalman_breathing_rate,
    kalman_breathing_rate_peaks,
) = (
    data["Timestamps Ventilator Volume Windowed Values"],
    data["Ventilator Volume Windowed Values"],
    data["Ventilator Rate Windowed Values"],
    data["Time in W"],
    data["V In W"],
    data["V Ex W"],
    data["Timestamps Intervals Volume"],
    data["Interval Volumes"],
    data["Timestamps Intervals Rate"],
    data["Kalman Breathing Rate"],
    data["Kalman Breathing Rate Peaks"],
)
plt.close()
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

axs[0, 0].plot(
    timestamps_ventilator_volume_windowed_values,
    ventilator_rate_windowed_values,
    color=colors[0],
    label="Ventilator",
)
# Plot camera respiratory rate
axs[0, 0].plot(
    timestamps_intervalls_rate, kalman_breathing_rate, label="Fourier", color=colors[1]
)
axs[0, 0].plot(
    timestamps_intervalls_rate,
    kalman_breathing_rate_peaks,
    label="Peaks",
    color=colors[2],
)
axs[0, 0].set_xlabel("Time (s)", fontsize=8)
axs[0, 0].set_ylabel("Rate (breaths/min)", fontsize=8)
axs[0, 0].set_title("(a) Respiratory rate", fontsize=8)

axs[0, 0].set_ylim(15, 80)
axs[0, 0].yaxis.set_ticks_position("left")
axs[0, 0].xaxis.set_ticks_position("bottom")
axs[0, 0].spines["right"].set_visible(False)
axs[0, 0].spines["left"].set_visible(False)
axs[0, 0].spines["top"].set_visible(False)
axs[0, 0].tick_params(axis="x", labelsize=8)
axs[0, 0].tick_params(axis="y", labelsize=8)
axs[0, 0].legend(loc="upper center", frameon=False, fontsize=8)


axs[1, 0].plot(
    timestamps_ventilator_volume_windowed_values,
    ventilator_volume_windowed_values,
    label="Ventilator",
    color=colors[0],
)

axs[1, 0].fill_between(
    time_in_w,
    v_in_w,
    v_ex_w,
    color=colors[0],
    alpha=0.4,
    label="Ventilator bounds",
)
# Plot camera estimates
axs[1, 0].plot(
    timestamps_intervalls_vol, intervall_volumes, label="Camera", color=colors[1]
)

axs[1, 0].set_xlabel("Time (s)", fontsize=8)
axs[1, 0].set_ylabel("Tidal volume (ml)", fontsize=8)
axs[1, 0].set_ylim(2, 9)
axs[1, 0].set_title("(b) Tidal volume", fontsize=8)
axs[1, 0].yaxis.set_ticks_position("left")
axs[1, 0].xaxis.set_ticks_position("bottom")
axs[1, 0].spines["right"].set_visible(False)
axs[1, 0].spines["left"].set_visible(False)
axs[1, 0].spines["top"].set_visible(False)
axs[1, 0].tick_params(axis="x", labelsize=8)
axs[1, 0].tick_params(axis="y", labelsize=8)
axs[1, 0].legend(loc="upper right", frameon=False, fontsize=8)

vital_sign = "Resp_Fourier"
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


print("Number of subjects: ", subject_number)
camera=np.asarray(camera)
truth=np.asarray(truth)
import BA_plot

# BA plot
BA_plot.mean_diff_plot(camera, truth, colors[0], ax=axs[0, 1])
axs[0, 1].set_ylabel("Difference (breaths/min)", fontsize=8)
axs[0, 1].set_xlabel("Means (breaths/min)", fontsize=8)
axs[0, 1].set_title("(c) Respiratory rate BA", fontsize=8)
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
axs[0, 2].set_xlabel("Camera - Ground Truth (breaths/min)", fontsize=8)
axs[0, 2].set_ylabel("Frequency", fontsize=8)
axs[0, 2].set_title("(e) Respiratory rate errors", fontsize=8)
axs[0, 2].spines["right"].set_visible(False)
axs[0, 2].spines["left"].set_visible(False)
axs[0, 2].spines["top"].set_visible(False)
axs[0, 2].yaxis.set_ticks_position("left")
axs[0, 2].xaxis.set_ticks_position("bottom")
axs[0, 2].tick_params(axis="x", labelsize=8)
axs[0, 2].tick_params(axis="y", labelsize=8)


# Correlation plot with linear regression
spacing = 60
axs[1, 1].scatter(truth[::spacing], camera[::spacing], color=colors[0], s=2.5)
axs[1, 1].set_ylabel("Camera (breaths/min)", fontsize=8)
axs[1, 1].set_xlabel("Ground Truth (breaths/min)", fontsize=8)
axs[1, 1].set_title("(d) Respiratory rate correlation", fontsize=8)
axs[1, 1].set_xlim(min(min(truth), min(camera)) - 10, max(max(truth), max(camera)) + 10)
axs[1, 1].set_ylim(min(min(truth), min(camera)) - 10, max(max(truth), max(camera)) + 10)
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
axs[1, 2].set_xlabel("Absolute error (breaths/min)", fontsize=8)
axs[1, 2].set_ylabel("Coverage probability (%)", fontsize=8)
axs[1, 2].set_title("(f) Respiratory rate CP", fontsize=8)
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
