
import sys

# add the folder mmerkat_analysis_pipeline
sys.path.insert(
    0,
    r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\meerkat_analysis_pipeline",
)

import RespRateAndVolumePipeline
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import MeerkatPipelineHelperfunctions
import numpy as np
import SummaryStatistics
import BA_plot

database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"

Resp = RespRateAndVolumePipeline.Respiratory_rate_and_volume_pipeline()
Resp.data_analysis_folder = database_path
Resp.run()
plt.close()

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
    kalman_breathing_rate_peaks
) = Resp.return_data()

import sys

# add the folder mmerkat_analysis_pipeline
sys.path.insert(
    0,
    r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\meerkat_analysis_pipeline",
)

import HeartRatePipeline
import matplotlib.pyplot as plt
import seaborn as sns
import BA_plot
import os
import MeerkatPipelineHelperfunctions
import numpy as np
import SummaryStatistics

database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"

hr = HeartRatePipeline.Calculate_heart_rate()
hr.data_analysis_folder = database_path
hr.run()
#7 for example

ECG_timestamps, ECG_rate, POS_kalman, CHROM_kalman, POS_kalman_peaks, CHROM_kalman_peaks, timestamps, average=hr.return_data()


import sys

# add the folder mmerkat_analysis_pipeline
sys.path.insert(
    0,
    r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\meerkat_analysis_pipeline",
)

import HeartRatePipeline
import matplotlib.pyplot as plt
import seaborn as sns
import BA_plot
import os
import MeerkatPipelineHelperfunctions
import numpy as np
import SummaryStatistics
import PulseOxygenationPipeline

database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"

ox = PulseOxygenationPipeline.CalculatePulseOxygenation()
ox.intervall_length = 900
ox.data_analysis_folder = database_path
ox.run()
# 5 for image

pulse_oximeter_timestamps, pulse_oximeter_spO2, time_spO2, spO2_calibrated, spO2_rgb, spO2_infrared, spO2_ycgcr, timestamps_spO2_calibrated=ox.return_data()



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
        timestamps_ventilator_volume_windowed_values,
        ventilator_rate_windowed_values,
        color=colors[0],
        label="Ventilator",
    )
# Plot camera respiratory rate
axs[0,0].plot(
    timestamps_intervalls_rate, kalman_breathing_rate, label="Camera", color=colors[1]
)
axs[0,0].set_xlabel("Time (s)", fontsize=8)
axs[0,0].set_ylabel("Rate (breaths/min)", fontsize=8)
axs[0,0].set_title("(a) Respiratory rate", fontsize=8)

axs[0,0].set_ylim(15, 80)
axs[0,0].yaxis.set_ticks_position("left")
axs[0,0].xaxis.set_ticks_position("bottom")
axs[0,0].spines["right"].set_visible(False)
axs[0,0].spines["left"].set_visible(False)
axs[0,0].spines["top"].set_visible(False)
axs[0,0].tick_params(axis='x', labelsize=8)
axs[0,0].tick_params(axis='y', labelsize=8)
axs[0,0].legend(loc='upper center', frameon=False, fontsize=8)



axs[1,0].plot(
timestamps_ventilator_volume_windowed_values,
ventilator_volume_windowed_values,

label="Ventilator",color=colors[0]
)

axs[1,0].fill_between(
    time_in_w,
    v_in_w,
    v_ex_w,
    color=colors[0],
    alpha=0.4,
    label="Ventilator bounds",
)
# Plot camera estimates
axs[1,0].plot(
timestamps_intervalls_vol,
intervall_volumes,
label="Camera",color=colors[1]
)

axs[1,0].set_xlabel("Time (s)", fontsize=8)
axs[1,0].set_ylabel("Tidal volume (ml)", fontsize=8)
axs[1,0].set_ylim(2, 9)
axs[1,0].set_title("(d) Tidal volume", fontsize=8)
axs[1,0].yaxis.set_ticks_position("left")
axs[1,0].xaxis.set_ticks_position("bottom")
axs[1,0].spines["right"].set_visible(False)
axs[1,0].spines["left"].set_visible(False)
axs[1,0].spines["top"].set_visible(False)
axs[1,0].tick_params(axis='x', labelsize=8)
axs[1,0].tick_params(axis='y', labelsize=8)
axs[1,0].legend(loc='upper right', frameon=False, fontsize=8)


axs[0,1].plot(ECG_timestamps, ECG_rate, label="ECG", color=colors[0])
axs[0,1].plot(
    timestamps, average, label="Camera", color=colors[1]
)
axs[0,1].set_ylabel("Heart rate (bpm)", fontsize=8)
axs[0,1].set_xlabel("Time (s)", fontsize=8)
axs[0,1].set_ylim(100, 200)
axs[0,1].set_title("(b) Heart rate", fontsize=8)
axs[0,1].legend(loc="lower center", fontsize=8, frameon=False)
axs[0,1].spines["right"].set_visible(False)
axs[0,1].spines["left"].set_visible(False)
axs[0,1].spines["top"].set_visible(False)
axs[0,1].yaxis.set_ticks_position("left")
axs[0,1].xaxis.set_ticks_position("bottom")
axs[0,1].tick_params(axis="x", labelsize=8)
axs[0,1].tick_params(axis="y", labelsize=8)



axs[0,2].plot(
            pulse_oximeter_timestamps,
            pulse_oximeter_spO2,
            label="Pulse oximeter",
            color=colors[0],
        )
axs[0,2].plot(time_spO2, spO2_infrared, label="Camera", color=colors[1])
axs[0,2].set_ylabel("SpO2 (%)", fontsize=8)
axs[0,2].set_xlabel("Time (s)", fontsize=8)
axs[0,2].set_title("(c) Pulse oximeter", fontsize=8)
axs[0,2].set_ylim(68, 100)
axs[0,2].legend(loc="lower right", fontsize=8, frameon=False)
axs[0,2].spines["right"].set_visible(False)
axs[0,2].spines["left"].set_visible(False)
axs[0,2].spines["top"].set_visible(False)
axs[0,2].yaxis.set_ticks_position("left")
axs[0,2].xaxis.set_ticks_position("bottom")
axs[0,2].tick_params(axis="x", labelsize=8)
axs[0,2].tick_params(axis="y", labelsize=8)



#for j in range(len(camera_loops_flow)):
#12    axs[0,0].plot(camera_loops_volume[j], camera_loops_flow[j], color=colors[j % 10])
                
        # Define plot parameters
axs[1,1].set_xlabel("Volume (ml)", fontsize=8)
axs[1,1].set_ylabel("Flow (ml/s)", fontsize=8)
axs[1,1].set_title("(e) Camera FV", fontsize=8)
axs[1,1].set_xlim(-10, 1)
axs[1,1].set_ylim(-60, 60)
axs[1,1].grid()
axs[1,1].spines["right"].set_visible(False)
axs[1,1].spines["left"].set_visible(False)
axs[1,1].spines["top"].set_visible(False)
axs[1,1].yaxis.set_ticks_position("left")
axs[1,1].xaxis.set_ticks_position("bottom")
axs[1,1].tick_params(axis="x", labelsize=8)
axs[1,1].tick_params(axis="y", labelsize=8)



#for j in range(len(ventilator_loops_flow)):
#        axs[1,0].plot(ventilator_loops_volume[j], ventilator_loops_flow[j], color=colors[j % 10])
                    
            # Define plot parameters
axs[1,2].set_xlabel("Volume (ml)", fontsize=8)
axs[1,2].set_ylabel("Flow (ml/s)", fontsize=8)
axs[1,2].set_title("(f) Ventilator FV", fontsize=8)
axs[1,2].set_xlim(-10, 1)
axs[1,2].set_ylim(-60, 60)
axs[1,2].grid()
axs[1,2].spines["right"].set_visible(False)
axs[1,2].spines["left"].set_visible(False)
axs[1,2].spines["top"].set_visible(False)
axs[1,2].yaxis.set_ticks_position("left")
axs[1,2].xaxis.set_ticks_position("bottom")
axs[1,2].tick_params(axis="x", labelsize=8)
axs[1,2].tick_params(axis="y", labelsize=8)


fig.align_ylabels()
fig.align_xlabels()
plt.tight_layout(pad=1)
plt.show()
