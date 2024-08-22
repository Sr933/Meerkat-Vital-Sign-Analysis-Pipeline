import numpy as np
import os
import MeerkatPipelineHelperfunctions 
import csv

class MeerkatStatisticalAnalysis:
    def __init__(self):
        """
        Initialize the MeerkatStatisticalAnalysis class with optional parameters.
        
        Parameters:
        - ground_truth_signal: List of ground truth signal values.
        - reference_signal: List of reference signal values.
        - ground_truth_timestamps: List of timestamps for ground truth signals.
        - reference_timestamps: List of timestamps for reference signals.
        - kappa: Sensitivity parameter for statistical analysis.
        - subject_folder: Directory path for saving results.
        - vital_sign: Type of vital sign being analyzed.
        - tidalvolumeflag: Flag indicating if tidal volume data is included.
        - tidalvolumeupper: List of upper bounds for tidal volume.
        - tidalvolumelower: List of lower bounds for tidal volume.
        """
        self.ground_truth_signal = []
        self.reference_signal = []
        self.ground_truth_timestamps = []
        self.reference_timestamps = []
        self.kappa = 10
        self.subject_folder = ""
        self.vital_sign = ""
        self.tidalvolumeflag = False
        self.tidalvolumeupper = []
        self.tidalvolumelower = []

    def run(self):
        # Function to tie all functions in class together

        if self.tidalvolumeflag:
            self.ground_truth_signal = self.tidalvolumeupper
            self.match_signals()
            self.tidalvolumeupper_matched = self.matched_ground_truth_signal.copy()
            self.ground_truth_signal = self.tidalvolumelower
            self.match_signals()
            self.tidalvolumelower_matched = self.matched_ground_truth_signal.copy()
            self.tidal_volume_matching()
            self.matched_ground_truth_signal = self.truth_upper_lower

        else:
            self.match_signals()
            
        self.save_matched_signals()
        MeerkatPipelineHelperfunctions.mean_absolute_diff(
            self.matched_ground_truth_signal, self.matched_reference_signal
        )
        MeerkatPipelineHelperfunctions.mean_square_diff(
            self.matched_ground_truth_signal, self.matched_reference_signal
        )
        MeerkatPipelineHelperfunctions.coverage_probability(
            self.matched_ground_truth_signal, self.matched_reference_signal, self.kappa
        )
        self.kappa = 2 * self.kappa
        MeerkatPipelineHelperfunctions.coverage_probability(
            self.matched_ground_truth_signal, self.matched_reference_signal, self.kappa
        )

    def match_signals(self):
        if len(self.ground_truth_signal) >= len(self.reference_signal):
            # When ground truth has more data points than reference

            # Aggregate the values in windows
            self.matched_ground_truth_signal = [
                np.mean(self.ground_truth_signal[(self.ground_truth_timestamps >= self.reference_timestamps[i]) & (self.ground_truth_timestamps <= self.reference_timestamps[i + 1])])
                for i in range(len(self.reference_timestamps) - 1)
            ]
            self.matched_reference_signal = self.reference_signal[:-1]  # Reference signal values except the last one

        else:
            # Aggregate the values in windows
            self.matched_reference_signal = [
                np.mean(self.reference_signal[(self.reference_timestamps >= self.ground_truth_timestamps[i]) & (self.reference_timestamps <= self.ground_truth_timestamps[i + 1])])
                for i in range(len(self.ground_truth_timestamps) - 1)
            ]
            self.matched_ground_truth_signal = self.ground_truth_signal[:-1]  # Ground truth signal values except the last one

    def match_signals2(self):
        # match the signals sampled at different rates with small ofsetts by caculating averages in the higher sampled signal
        ground_truth_signal = np.array(self.ground_truth_signal)
        reference_signal = np.array(self.reference_signal)
        ground_truth_timestamps = np.array(self.ground_truth_timestamps)
        reference_timestamps = np.array(self.reference_timestamps)
        matched_ground_truth = []
        matched_reference = []
        if len(ground_truth_signal) >= len(reference_signal):
            for i in range(len(reference_signal) - 1):
                timestart = reference_timestamps[i]
                timeend = reference_timestamps[i + 1]
                value_indices_in_window = np.where(
                    np.logical_and(
                        ground_truth_timestamps >= timestart,
                        ground_truth_timestamps <= timeend,
                    )
                )[0]

                values_in_window = []
                if len(value_indices_in_window > 0):
                    for j in value_indices_in_window:
                        if j < len(ground_truth_signal):
                            values_in_window.append(ground_truth_signal[j])

                    matched_ground_truth.append(np.mean(np.array(values_in_window)))
                    matched_reference.append(
                        (reference_signal[i])
                    )  

        else :
            for i in range(len(ground_truth_signal) - 1):
                timestart = ground_truth_timestamps[i]
                timeend = ground_truth_timestamps[i + 1]
                value_indices_in_window = np.where(
                    np.logical_and(
                        reference_timestamps >= timestart,
                        reference_timestamps <= timeend,
                    )
                )[0]

                values_in_window = []
                if len(value_indices_in_window) > 0:
                    for j in value_indices_in_window:
                        if j < len(reference_signal):
                            values_in_window.append(reference_signal[j])
                    matched_reference.append(np.mean(np.array(values_in_window)))
                    matched_ground_truth.append(ground_truth_signal[i])

        self.matched_ground_truth_signal = np.array(matched_ground_truth)
        self.matched_reference_signal = np.array(matched_reference)  
        
    def save_matched_signals(self):
        output_folder = os.path.join(self.subject_folder, "Pipeline results")
        output_filepath = os.path.join(output_folder, self.vital_sign)
        
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        
        with open(output_filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Ground Truth Signal', 'Reference Signal'])
            
            # Write the data rows
            for gt_signal, ref_signal in zip(self.matched_ground_truth_signal, self.matched_reference_signal):
                writer.writerow([gt_signal, ref_signal])

    def tidal_volume_matching(self):

        # Create a mask for values within the bounds
        within_bounds_mask = (
            (self.matched_reference_signal >= self.tidalvolumelower_matched) &
            (self.matched_reference_signal <= self.tidalvolumeupper_matched)
        )

        # Initialize the output array
        self.truth_upper_lower = np.zeros_like(self.matched_reference_signal)

        # Assign values within bounds directly
        self.truth_upper_lower[within_bounds_mask] = self.matched_reference_signal[within_bounds_mask]

        # Calculate absolute differences for out-of-bounds values
        upper_bound_diff = np.abs(self.matched_reference_signal - self.tidalvolumeupper_matched)
        lower_bound_diff = np.abs(self.matched_reference_signal - self.tidalvolumelower_matched)

        # Determine the closest bound for out-of-bounds values
        closest_bound_mask = ~within_bounds_mask
        self.truth_upper_lower[closest_bound_mask] = np.where(
            upper_bound_diff[closest_bound_mask] > lower_bound_diff[closest_bound_mask],
            self.tidalvolumelower_matched[closest_bound_mask],
            self.tidalvolumeupper_matched[closest_bound_mask]
    )

