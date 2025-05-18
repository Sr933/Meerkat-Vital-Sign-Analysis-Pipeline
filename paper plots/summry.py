import os
import numpy as np

def vital_sign_summary_statistics(data_analysis_folder, vital_sign):
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

                        # Iterate over file
                        with open(signal_filepath) as f:
                            for line in f:
                                currentline = line.split(",")
                                t = float(currentline[0])
                                r = float(currentline[1][:-1])

                                truth.append(t)
                                camera.append(r)
                        subject_number += 1

        except:
            continue
    print("Number of subjects: ", subject_number)
    truth = np.array(truth)
    camera = np.array(camera)

    ##Summary statistics
    print(vital_sign)
    kappa = 10 if "saturation" not in vital_sign else 3
    if "POS" in vital_sign or "CHROM" in vital_sign:
        kappa=5
    MeerkatPipelineHelperfunctions.coverage_probability(truth, camera, kappa=kappa)
    kappa = kappa * 2
    MeerkatPipelineHelperfunctions.coverage_probability(truth, camera, kappa=kappa)
    MeerkatPipelineHelperfunctions.mean_absolute_diff(truth, camera)
    MeerkatPipelineHelperfunctions.mean_square_diff(truth, camera)