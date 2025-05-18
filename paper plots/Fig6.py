from MeerkatPipeline import (
    HeartRatePipeline,
    PulseOxygenationPipeline,
    FlowVolumeLoopPipeline,
    RespiratoryPipeline,
    BreathingAsymmetryPipeline,
)
import matplotlib.pyplot as plt
import seaborn as sns

database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"

### First 13 for ventilator comparison, then 5 for normal, finally 10 for abnormal


fv = FlowVolumeLoopPipeline(
    data_analysis_folder=database_path, plot_single_loops_flag=False
)
fv.run()

data = fv.return_data()
(
    ventilator_loops_flow1,
    ventilator_loops_volume1,
    camera_loops_flow1,
    camera_loops_volume1,
) = (
    data["Ventilator loops flow"],  # Flow data from ventilator loops.
    data["Ventilator loops volume"],  # Volume data from ventilator loops.
    data["Camera loops flow"],  # Flow data from camera loops.
    data["Camera loops volume"],
)  # Volume data from camera loops.

# 5 is example
# 3 as well


fv = FlowVolumeLoopPipeline(
    data_analysis_folder=database_path, plot_single_loops_flag=False
)
fv.run()

data = fv.return_data()
(
    ventilator_loops_flow2,
    ventilator_loops_volume2,
    camera_loops_flow2,
    camera_loops_volume2,
) = (
    data["Ventilator loops flow"],  # Flow data from ventilator loops.
    data["Ventilator loops volume"],  # Volume data from ventilator loops.
    data["Camera loops flow"],  # Flow data from camera loops.
    data["Camera loops volume"],)
# 5 is example
# 3 as well




fv = FlowVolumeLoopPipeline(
    data_analysis_folder=database_path, plot_single_loops_flag=False
)
fv.run()

data = fv.return_data()
(
    ventilator_loops_flow3,
    ventilator_loops_volume3,
    camera_loops_flow3,
    camera_loops_volume3,
) = (
    data["Ventilator loops flow"],  # Flow data from ventilator loops.
    data["Ventilator loops volume"],  # Volume data from ventilator loops.
    data["Camera loops flow"],  # Flow data from camera loops.
    data["Camera loops volume"],)

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
fig, axs = plt.subplots(2, 2, figsize=(8.27, 4.5))

colors = sns.color_palette("deep")

for j in range(len(camera_loops_flow1)):
    axs[0, 0].plot(camera_loops_volume1[j], camera_loops_flow1[j], color=colors[j % 10])

    # Define plot parameters
axs[0, 0].set_xlabel("Volume (ml)", fontsize=8)
axs[0, 0].set_ylabel("Flow (ml/s)", fontsize=8)
axs[0, 0].set_title("(a) Camera", fontsize=8)
axs[0, 0].set_xlim(-10, 1)
axs[0, 0].set_ylim(-60, 60)
axs[0, 0].grid()
axs[0, 0].spines["right"].set_visible(False)
axs[0, 0].spines["left"].set_visible(False)
axs[0, 0].spines["top"].set_visible(False)
axs[0, 0].yaxis.set_ticks_position("left")
axs[0, 0].xaxis.set_ticks_position("bottom")
axs[0, 0].tick_params(axis="x", labelsize=8)
axs[0, 0].tick_params(axis="y", labelsize=8)

for j in range(len(ventilator_loops_flow1)):
    axs[1, 0].plot(
        ventilator_loops_volume1[j], ventilator_loops_flow1[j], color=colors[j % 10]
    )

    # Define plot parameters
axs[1, 0].set_xlabel("Volume (ml)", fontsize=8)
axs[1, 0].set_ylabel("Flow (ml/s)", fontsize=8)
axs[1, 0].set_title("(b) Ventilator", fontsize=8)
axs[1, 0].set_xlim(-10, 1)
axs[1, 0].set_ylim(-60, 60)
axs[1, 0].grid()
axs[1, 0].spines["right"].set_visible(False)
axs[1, 0].spines["left"].set_visible(False)
axs[1, 0].spines["top"].set_visible(False)
axs[1, 0].yaxis.set_ticks_position("left")
axs[1, 0].xaxis.set_ticks_position("bottom")
axs[1, 0].tick_params(axis="x", labelsize=8)
axs[1, 0].tick_params(axis="y", labelsize=8)

###
colors = sns.color_palette("deep")

for j in range(len(camera_loops_flow2)):
    axs[0, 1].plot(camera_loops_volume2[j], camera_loops_flow2[j], color=colors[j % 10])

    # Define plot parameters
axs[0, 1].set_xlabel("Volume (ml)", fontsize=8)
axs[0, 1].set_ylabel("Flow (ml/s)", fontsize=8)
axs[0, 1].set_title("(c) Camera normal", fontsize=8)
axs[0, 1].set_xlim(-10, 1)
axs[0, 1].set_ylim(-60, 60)
axs[0, 1].grid()
axs[0, 1].spines["right"].set_visible(False)
axs[0, 1].spines["left"].set_visible(False)
axs[0, 1].spines["top"].set_visible(False)
axs[0, 1].yaxis.set_ticks_position("left")
axs[0, 1].xaxis.set_ticks_position("bottom")
axs[0, 1].tick_params(axis="x", labelsize=8)
axs[0, 1].tick_params(axis="y", labelsize=8)


for j in range(len(camera_loops_flow3)):
    axs[1, 1].plot(camera_loops_volume3[j], camera_loops_flow3[j], color=colors[j % 10])

    # Define plot parameters
axs[1, 1].set_xlabel("Volume (ml)", fontsize=8)
axs[1, 1].set_ylabel("Flow (ml/s)", fontsize=8)
axs[1, 1].set_title("(d) Camera abnormal", fontsize=8)
axs[1, 1].set_xlim(-10, 1)
axs[1, 1].set_ylim(-60, 60)
axs[1, 1].grid()
axs[1, 1].spines["right"].set_visible(False)
axs[1, 1].spines["left"].set_visible(False)
axs[1, 1].spines["top"].set_visible(False)
axs[1, 1].yaxis.set_ticks_position("left")
axs[1, 1].xaxis.set_ticks_position("bottom")
axs[1, 1].tick_params(axis="x", labelsize=8)
axs[1, 1].tick_params(axis="y", labelsize=8)


fig.align_ylabels()
fig.align_xlabels()
fig.tight_layout(pad=1)
plt.show()


