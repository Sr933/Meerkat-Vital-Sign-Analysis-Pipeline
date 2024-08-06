import numpy as np
import os
import MeerkatPipelineHelperfunctions


class MeerkatStatisticalAnalysis:
    def __init__(self):
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
        print("Data points:", len(self.matched_ground_truth_signal))

    def save_matched_signals(self):
        output_folder = repr("\\" + "Pipeline results")
        output_folder = self.subject_folder + output_folder[2:-1]
        output_file = repr("\\" + self.vital_sign)
        output_filepath = output_folder + output_file[2:-1]
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        with open(output_filepath, "w") as f:
            for i in range(len(self.matched_ground_truth_signal)):
                output = (
                    str(self.matched_ground_truth_signal[i])
                    + ","
                    + str(self.matched_reference_signal[i])
                )
                f.write(output)
                f.write("\n")

    def tidal_volume_matching(self):
        self.truth_upper_lower = np.zeros(len(self.matched_reference_signal))
        for i in range(len(self.matched_reference_signal)):
            if (
                self.matched_reference_signal[i] >= self.tidalvolumelower_matched[i]
                and self.matched_reference_signal[i] <= self.tidalvolumeupper_matched[i]
            ):
                self.truth_upper_lower[i] = self.matched_reference_signal[i]
            else:
                upper_bound = abs(
                    self.matched_reference_signal[i] - self.tidalvolumeupper_matched[i]
                )
                lower_bound = abs(
                    self.matched_reference_signal[i] - self.tidalvolumelower_matched[i]
                )
                if upper_bound > lower_bound:
                    self.truth_upper_lower[i] = self.tidalvolumelower_matched[i]
                else:
                    self.truth_upper_lower[i] = self.tidalvolumeupper_matched[i]
