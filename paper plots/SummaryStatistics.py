import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import MeerkatPipelineHelperfunctions

from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy


def ground_truth_characterisation(data_analysis_folder):
    # Iterate over dataset to obtain all patients
    subjects = []
    for file in os.listdir(data_analysis_folder):
        if "mk" in file:
            subjects.append(os.fsdecode(file))

    heart_rate = []
    oxygen_saturation = []
    respiratory_rate = []
    tidal_volumes = []

    # Iterate over all subjects
    for subject in subjects:
        # Define subject folder
        file_folder = repr("\\" + subject)
        file_folder = file_folder[2:-1]
        subject_folder = data_analysis_folder + file_folder

        # Define heart rate folder
        hr_folder = repr("\\" + "ECG heart rate")
        hr_folder = subject_folder + hr_folder[2:-1]

        # Check if file exists and import if required
        if len(os.listdir(hr_folder)) > 0:
            hr_folder = repr("\\" + "ECG heart rate")
            hr_folder = subject_folder + hr_folder[2:-1]
            hr_file = repr("\\" + os.listdir(hr_folder)[0])
            hr_filepath = hr_folder + hr_file[2:-1]
            df = pd.read_csv(hr_filepath)
            hr = np.array(df[" Heart rate (bpm)"])
            for i in range(len(hr)):
                heart_rate.append(int(hr[i]))

        # Define oxygen saturation folder
        oximeter_folder = repr("\\" + "Pulse oximeter oxygen saturation")
        oximeter_folder = subject_folder + oximeter_folder[2:-1]

        # Check if file exists and import if required
        if len(os.listdir(oximeter_folder)) > 0:
            oximeter_file = os.listdir(oximeter_folder)[0]
            oximeter_file = repr("\\" + oximeter_file)
            oximeter_filepath = oximeter_folder + oximeter_file[2:-1]
            df = pd.read_csv(oximeter_filepath)
            pulse_oximeter_spO2 = np.array(df[" Oxygen saturation (%)"])
            for i in range(len(pulse_oximeter_spO2)):
                oxygen_saturation.append(int(pulse_oximeter_spO2[i]))

        # Define resp rate folder
        rate_folder = repr("\\" + "Ventilator respiratory rate")
        rate_folder = subject_folder + rate_folder[2:-1]

        # Check if file exists
        if len(os.listdir(rate_folder)) > 0:
            # Import resp rate
            rate_file = os.listdir(rate_folder)[0]
            rate_file = repr("\\" + rate_file)
            rate_filepath = rate_folder + rate_file[2:-1]
            df = pd.read_csv(rate_filepath)
            ventilator_rate = np.array(df[" Respiratory rate (breaths/min)"])
            for i in range(len(pulse_oximeter_spO2)):
                respiratory_rate.append(int(ventilator_rate[i]))

            # Import tidal volumes
            volume_folder = repr("\\" + "Ventilator tidal volume best estimate")
            volume_folder = subject_folder + volume_folder[2:-1]
            camera_file = os.listdir(volume_folder)[0]
            camera_file = repr("\\" + camera_file)
            camera_filepath = volume_folder + camera_file[2:-1]
            df2 = pd.read_csv(camera_filepath)
            ventilator_tidal_volume = np.array(df2[" VT (ml)"])
            for i in range(len(pulse_oximeter_spO2)):
                tidal_volumes.append(float(ventilator_tidal_volume[i]))

    # Plot all imported data
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
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8.27, 2))
   
    colors = sns.color_palette("deep")
    plt.subplots_adjust(
        left=0.07, top=0.9, right=0.93, bottom=0.15, hspace=0.2, wspace=0.5
    )
    ax1.hist(heart_rate, bins=range(min(heart_rate), max(heart_rate) + 1, 1), color=colors[0])
    ax1.set_xlabel("Rate (beats/min)", fontsize=8)
    ax1.set_ylabel("Frequency", fontsize=8)
    ax1.set_title("(a) Heart rate", fontsize=8)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    
    ax2.hist(
        oxygen_saturation,
        bins=range(min(oxygen_saturation), max(oxygen_saturation) + 1, 1), color=colors[0]
    )
    ax2.set_xlabel("Saturation (%)", fontsize=8)
    ax2.set_ylabel("Frequency", fontsize=8)
    ax2.set_title("(b) Oxygen saturation", fontsize=8)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_ticks_position("left")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    
    ax3.hist(
        respiratory_rate,
        bins=range(int(min(respiratory_rate)), int(max(respiratory_rate)) + 1, 1), color=colors[0]
    )
    ax3.set_xlabel("Rate (breaths/min)", fontsize=8)
    ax3.set_ylabel("Frequency", fontsize=8)
    ax3.set_title("(c) Respiratory rate", fontsize=8)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.yaxis.set_ticks_position("left")
    ax3.xaxis.set_ticks_position("bottom")
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    
    
    ax4.hist(
        tidal_volumes,
        bins=np.arange(min(tidal_volumes), max(tidal_volumes) + 0.25, 0.25), color=colors[0]
    )
    ax4.set_xlabel("Volume (ml)", fontsize=8)
    ax4.set_ylabel("Frequency", fontsize=8)
    ax4.set_title("(d) Tidal volume", fontsize=8)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.yaxis.set_ticks_position("left")
    ax4.xaxis.set_ticks_position("bottom")
    ax4.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    plt.tight_layout(pad=1)
    plt.show()





def coverage_probability_abs(truth, test, kappa):
    # Calculate proportion of values lying between the upper and lower acceptbale bounds
    truth = np.array(truth)
    truth_upper = truth + kappa
    truth_lower = truth - kappa
    count_in = 0
    for i in range(len(truth)):
        if test[i] > truth_lower[i] and test[i] < truth_upper[i]:
            count_in += 1

    tdi = count_in / len(truth)
    return tdi


def tidal_volume_outcome_summary(file):
    # Read data
    data = pd.read_csv(file)
    
    
    
    volumes_ox=[]
    volumes_no_ox=[]
    volumes_ox_corrected=[]
    volumes_no_ox_corrected=[]

    n=len(data["ID"])
    for i in range(n):
        vol=data["Tidal volume (ml)"][i]
        weight=data["Recording Weight (g)"][i]
        if data["Breathing @ 36 weeks"][i]=="Normal":  
            volumes_no_ox.append(vol)
            volumes_no_ox_corrected.append(vol/weight)
        elif data["Breathing @ 36 weeks"][i]=="Poor":
            volumes_ox.append(vol)
            volumes_ox_corrected.append(vol/weight)
        else:
            volumes_no_ox.append(vol)
            volumes_no_ox_corrected.append(vol/weight)
        
    print("Raw: ",scipy.stats.mannwhitneyu(volumes_ox, volumes_no_ox))
    print("Corrected: ", scipy.stats.mannwhitneyu(volumes_ox_corrected, volumes_no_ox_corrected))

    # Plot raw tidal volumes
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.style.use(['default'])
    colors = sns.color_palette("deep")
    sns.swarmplot(
        ax=ax1, data=data, x="Breathing @ 36 weeks", y="Tidal volume (ml)", size=5, color=colors[0]
    )
    sns.boxplot(
        ax=ax1,
        data=data,
        x="Breathing @ 36 weeks",
        y="Tidal volume (ml)",
        fill=True,
        showfliers=False,
        color=".8",
    )
    ax1.set_xlabel("Breathing at 36 weeks", fontsize=20)
    ax1.set_ylabel("Tidal volume (ml)", fontsize=20)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.set_title("Raw tidal volumes", fontsize=20)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    # Plot corrected tidal volumes
    sns.swarmplot(
        ax=ax2,
        data=data,
        x="Breathing @ 36 weeks",
        y="Corrected tidal volume (ml/kg)",color=colors[0]
    )
    sns.boxplot(
        ax=ax2,
        data=data,
        x="Breathing @ 36 weeks",
        y="Corrected tidal volume (ml/kg)",
        fill=True,
        showfliers=False,
        color=".8",
    )
    ax2.set_xlabel("Breathing at 36 weeks", fontsize=20)
    ax2.set_ylabel("Corrected tidal volume (ml/kg)", fontsize=20)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_ticks_position("left")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.set_title("Standardised tidal volumes", fontsize=20)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)


    plt.tight_layout(pad=2.5,w_pad=2.5)
    plt.show()
