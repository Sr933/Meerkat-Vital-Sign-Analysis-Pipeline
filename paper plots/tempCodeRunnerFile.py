import sys

# add the folder mmerkat_analysis_pipeline
sys.path.insert(
    0,
    r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\meerkat_analysis_pipeline",
)
import pandas as pd
import RespRateAndVolumePipeline
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import MeerkatPipelineHelperfunctions
import numpy as np
import SummaryStatistics

database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"


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

file=r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\Clinical outcomes\Volumes_and_clinical_outcomes.csv"
data = pd.read_csv(file)

sns.swarmplot(
        ax=axs[0,0], data=data, x="Breathing @ 36 weeks", y="Tidal volume (ml)", size=2.5, color=colors[0]
    )
sns.boxplot(
    ax=axs[0,0],
    data=data,
    x="Breathing @ 36 weeks",
    y="Tidal volume (ml)",
    fill=True,
    showfliers=False,
    color=".8",
)
axs[0,0].set_xlabel("Breathing at 36 weeks", fontsize=8)
axs[0,0].set_ylabel("Volume (ml)", fontsize=8)
axs[0,0].spines["right"].set_visible(False)
axs[0,0].spines["left"].set_visible(False)
axs[0,0].spines["top"].set_visible(False)
axs[0,0].yaxis.set_ticks_position("left")
axs[0,0].xaxis.set_ticks_position("bottom")
axs[0,0].set_title("(a) Raw tidal volumes", fontsize=8)
axs[0,0].tick_params(axis='x', labelsize=8)
axs[0,0].tick_params(axis='y', labelsize=8)
# Plot corrected tidal volumes
sns.swarmplot(
    ax=axs[1,0],
    data=data,
    x="Breathing @ 36 weeks",
    y="Corrected tidal volume (ml/kg)",color=colors[0], size=2.5
)