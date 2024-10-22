#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:12:50 2020

@author: David Chong Tian Wei
"""


import json
import datetime
import math
import numpy as np
import pandas as pd
from atpbar import register_reporter, find_reporter, atpbar
import enum
class fs(enum.Enum):
    """
    Enumeration for flow states
    
    Attributes
    ----------
    no_flow : int 
        0
    inspiration_initiation : int
        1
    peak_inspiratory_flow : int
        2
    inspiration_termination : int
        3
    expiration_initiation : int
        4
    peak_expiratory_flow : int
        5
    expiration_termination : int
        6
    """
    no_flow = 0
    inspiration_initiation = 1
    peak_inspiratory_flow = 2
    inspiration_termination = 3
    expiration_initiation = 4
    peak_expiratory_flow = 5
    expiration_termination = 6
    
class ps(enum.Enum):
    """
    Enumeration for pressure states
    
    Attributes
    ----------
    peep : int
        0
    pressure_rise : int
        1
    pip : int
        2
    pressure_drop : int
        3
    """
    peep = 0
    pressure_rise = 1
    pip = 2
    pressure_drop = 3
    
class GeneralPipeline:
    """ 
    Utility class to help tie the different parts of the package together into an easy to use pipeline
    
    Attributes
    ----------
    data : Pandas Dataframe
        The record to be analyzed. Populated using load_data
    mapper : StateMapper object
        Used for performing the state mapping phase
    labeller : PhaseLabeller object
        Used for peforming the segmentation and sub-phase labelling steps. Also will contain breaths after processing.
    config : Dictionary
        Used to store configuration details for logging
    configured : Boolean
        Whether configure has been called on the object
    data_loaded : Boolean
        Whether load_data has been called on the object
    """
    def __init__(self):
        """
        Initialises the pipeline with placeholder objects
        """
        self.data = None
        self.mapper = StateMapper()
        self.labeller = PhaseLabeller()
        self.config = {}
        self.configured = False
        self.data_loaded = False

    def load_data(self, path, cols):
        """
        Loads the data specified by path and cols and performs linear interpolation with window average baseline correction
        
        Parameters
        ----------
        path : string
            Path to the data file
        cols : array like of int
            Columns in the data file corresponding to time, pressure, and flow respectively
        
        Returns
        -------
        None
        """
        if not self.configured:
            print("Please configure the pipeline first")
            return
        self.data = pre_process_ventilation_data(path, cols)
        if self.config["correction_window"] is not None:
            correct_baseline(self.data.iloc[:,2], self.config["correction_window"])
        self.data.iloc[:,2] = self.flow_unit_converter(self.data.iloc[:,2])
        self.config["input_file"] = path
        self.data_loaded = True
        
    def configure(self,correction_window=None, flow_unit_converter=lambda x:x,
                  freq=100, peep=5.5, flow_thresh=0.1, t_len=0.03, f_base=0,
                  leak_perc_thresh=0.66, permit_double_cycling=False,
                  insp_hold_length=0.5, exp_hold_length=0.05):
        """ 
        Overall coniguration for the pipeline. Please call before process and load data
        
        Parameters
        ----------
        correction_window : int, optional
            Size of the window to perform baseline correction by centering on average. Defaults to None (no correction).
        flow_unit_converter : f: real -> real, optional
            Function to convert units of flow and flow_threshold to desired units to be displayed. Defaults to the identity function.
        freq : int, optional
            Sampling rate of the sample being analyzed. Defaults to 100
        peep : real, optional
            The value which will be considered baseline pressure. Defaults to 5.5
        flow_thresh : real, optional
            The minimum threshold that flow must cross to be considered a new breath. Defaults to 0.1
        t_len : real, optional
            Length of the window in seconds to perform state mapping. Defaults to 0.03
        f_base : real, optional
            Value for flow to be considered no_flow. Defaults to 0
        leak_perc_thresh : real, optional
            Maximum percentage difference between inspiratory and expiratory volume for a breath to be considered normal. Defaults to 66%.
        permit_double_cycling : boolean, optional
            Flag to decide whether to mergre double cycles. Defaults to false.
        insp_hold_length : real
            Maximum time in seconds from inspiration until an expiration is encountered, after which the breath is terminated. Defaults to 0.5
        exp_hold_length : real
            Maximum expiratory hold length between breaths to be considered double cycling. Defaults to 0.05s
        
        Returns
        -------
        None
        """
        self.config["correction_window"] = correction_window
        self.flow_unit_converter = flow_unit_converter
        self.config["freq"] = freq
        self.config["peep"] = peep
        self.config["flow_thresh"] = flow_unit_converter(flow_thresh)
        self.config["f_base"] = f_base
        self.config["t_len"] = t_len
        self.config["leak_perc_thresh"] = leak_perc_thresh
        self.config["permit_double_cycling"] = permit_double_cycling
        self.config["insp_hold_length"] = insp_hold_length
        self.config["exp_hold_length"] = exp_hold_length
        
        self.configured = True
        
    def process(self, log=True, output_files=True):
        """
        Processes the data after configuration and loading of data
        
        Parameters
        ----------
        log : boolean, optional
            Flag to decide whether to create output logs for the analysis. Defaults to True
        output_files : boolean, optional
            Flag to decide whether to create output files. Defaults to True
        
        Returns
        -------
        None
        """
        if not self.configured:
            print("Please configure the pipeline first")
            return
        if not self.data_loaded:
            print("Please load data first")
            return
        self.config["processing_start_time"] = datetime.datetime.now()
        
        self.mapper.configure(p_base=self.config["peep"],f_base=self.config["f_base"], 
                              f_thresh=self.config["flow_thresh"],freq=self.config["freq"],
                              t_len=self.config["t_len"])
        self.labeller.configure(freq=self.config["freq"],
                                hold_length=self.config["insp_hold_length"],
                                leak_perc_thresh=self.config["leak_perc_thresh"],
                                permit_double_cycling=self.config["permit_double_cycling"],
                                exp_hold_len=self.config["exp_hold_length"])
        self.mapper.process(self.data.iloc[:,1], self.data.iloc[:,2])
        self.labeller.process(self.mapper.p_labels, self.mapper.f_labels,
                              self.data.iloc[:,1], self.data.iloc[:,2])
        self.config["processing_end_time"] = datetime.datetime.now()
        self.config["time_elapsed"] = str(self.config["processing_end_time"] - self.config["processing_start_time"])
        stem = ".".join(self.config["input_file"].split(".")[:-1])
        if output_files:
            breaths_raw = self.labeller.get_breaths_raw()
            breaths_raw["max_expiratory_flow"] = breaths_raw["max_expiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths_raw["max_inspiratory_flow"] = breaths_raw["max_inspiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths_raw.to_csv(stem + "_predicted_Breaths_Raw.csv", index=False)
            breaths = self.labeller.get_breaths()
            breaths["max_expiratory_flow"] = breaths["max_expiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths["max_inspiratory_flow"] = breaths["max_inspiratory_flow"].apply(lambda x : x / self.flow_unit_converter(1))
            breaths.to_csv(stem + "_predicted_Breaths_ms.csv", index=False)
            self.mapper.get_labels().to_csv(stem + "_predicted_Pressure_And_Flow_States.csv", index=False)
            self.labeller.get_breath_annotations(self.data.shape[0]).to_csv(stem + "_predicted_Breaths_Annotations.csv", index=False)
            self.config["output_files"] = [stem + "_predicted_Breaths_Raw.csv",
                                           stem + "_predicted_Breaths_ms.csv",
                                           stem + "_predicted_Pressure_And_Flow_States.csv",
                                           stem + "_predicted_Breaths_Annotations.csv"]
        if log:
            self.config["processing_start_time"] = str(self.config["processing_start_time"])
            self.config["processing_end_time"] = str(self.config["processing_end_time"])
            f = open(stem + "_run_config.json","w")
            f.write(json.dumps(self.config))
            f.close()

class StateMapper:
    """Class for mapping pressure and flow time series to enumerated states.
    
    Attributes
    ----------
    p_base : real
        Baseline pressure to be used in processing. Usually can be found from ventilator settings
    f_base : real
        Baseline flow to be used in processing. Is usually set to 0
    f_thresh : real
        Threshold which flow must be larger than in order to be considered a moving state
    freq : real
        Sampling rate for the record to be processed in Hz
    w_len : integer
        Number of datapoints to be used in a window calculation
    p_labels : Numpy array of PressureStates enum
        Contains the calculated pressure state labels after process is called
    f_labels : Numpy array of FlowStates enum
        Contains the calculated flow state labels after process is called
    """
    
    def __init__(self):
        self.p_labels = np.array([])
        self.f_labels = np.array([])
        self.configure();
    
    def configure(self, p_base=5.5, f_base=0, f_thresh=0.1, freq=100, t_len=0.03):
        """ 
        Sets processing constants. To be called before process
        
        Parameters
        ----------
        p_base : real, optional
            Baseline pressure / PEEP set on the ventilator. Defaults to 5.5
        f_base : real, optional
            Baseline flow which is usually 0. Defaults to 0
        f_thresh : real, optional
            Threshold for the standard deviation of a window to be considered non-stationary. Defaults to 0.1
        freq : int, optional
            Sampling rate of the input to be processed. Defaults to 100
        t_len : real, optional
            Length of time in seconds of the desired window to use. Defaults to 0.03s
        
        Returns
        -------
        None
        """
        self.p_base = p_base
        self.f_base = f_base
        self.f_thresh = f_thresh
        self.freq = freq
        self.w_len = math.ceil(freq * t_len)
        if self.w_len < 3:
            print("Warning: calculated window length is less than 3, average and standard deviation calculations may be unhelpful, consider increasing t_len")
    
    def get_labels(self):
        """
        Returns the calculated labels
        
        Returns
        -------
        Pandas dataframe
            Dataframe containing the calculated pressure and flow states as integers
        """
        return pd.DataFrame({"Pressure_States" : [x.value for x in self.p_labels], "Flow_States" : [x.value for x in self.f_labels]})
            
    def process_pressures(self, pressures, p_0=ps.peep, con=None, reporter=None):
        """
        Maps data points from pressure to enumerated states
        
        Parameters
        ----------
        pressures : array like of real
            Pressure data points
        p_0 : PressureStates enum, optional
            The initial pressure state the program assumes it is in. Defaults to peep
        con : Queue object, optional
            The queue object to use for transferring data back to main cpu if multiprocessing is used. Defaults to None
        reporter : reporter object, optional
            The reporter object used to update the main cpu on the progress of the analysis. Defaults to None
        
        Returns
        -------
        None
        """
        if len(pressures) < self.w_len and con is None:
            con.put(np.array([]))
        elif len(pressures) < self.w_len:
            return
        
        output = np.array([ps(p_0)] * len(pressures))
        pressures = np.array(pressures).reshape(-1, self.w_len)
        pressure_means = np.mean(pressures, axis=1)
        pressure_mean_deltas = pressure_means[1:] - pressure_means[:-1]
        pressure_stds = np.std(pressures, axis=1)
        prev_label = ps(p_0)
        prev_pressure_hold = self.p_base
        if reporter != None:
            register_reporter(reporter)
        for i in atpbar(range(len(pressure_means)-1), name="Labelling pressure states"):
            if pressure_stds[i] < 0.1 * self.p_base:
                w_std_i = 0.05 * self.p_base
            else:
                w_std_i = pressure_stds[i]
            w_mean_delta = pressure_mean_deltas[i]
            w_mean_i_1 = pressure_means[i+1]
            curpos = (i+1) * self.w_len
            # If standard deviation is too small, set it to a minimum threshold
            if w_std_i < 0.1 * self.p_base:
                w_std_i = self.p_base * 0.05
            
            # Process stationary states
            if abs(w_mean_delta) < 2 * w_std_i:
                # Process for PIP
                if w_mean_i_1 > (self.p_base + prev_pressure_hold) / 2 + 2 * w_std_i:
                    if prev_label is not ps.pip:
                        output[curpos:curpos + self.w_len] = ps.pip
                        prev_pressure_hold = pressure_means[i+1]
                        prev_label = ps.pip
                    else:
                        output[curpos:curpos + self.w_len] = ps.pip
                        prev_label = ps.pip
                else:
                    # Process PEEP
                    if prev_label is not ps.peep:
                        output[curpos:curpos + self.w_len] = ps.peep
                        prev_pressure_hold = pressure_means[i+1]
                        prev_label = ps.peep
                    else:
                        output[curpos:curpos + self.w_len] = ps.peep
                        prev_label = ps.peep
            elif w_mean_delta > 0:
                # Process pressure rise
                output[curpos:curpos + self.w_len] = ps.pressure_rise
                prev_label = ps.pressure_rise
            else:
                # Process pressure drop
                output[curpos:curpos + self.w_len] = ps.pressure_drop
                prev_label = ps.pressure_drop
        if con is not None:
            con.put(output)
        else:
            self.p_labels = np.concatenate([self.p_labels, output])
            
    def process_flows(self, flows, f_0=fs.no_flow, con=None, reporter=None):
        """
        Maps data points from pressure to enumerated states
        
         Parameters
        ----------
        flows : array like of real
            Flow data points
        f_0 : FlowStates enum, optional
            The initial flow state the program assumes it is in. Defaults to no flow.
        con : Queue object, optional
            The queue object to use for transferring data back to main cpu if multiprocessing is used. Defaults to None
        reporter : reporter object, optional
            The reporter object used to update the main cpu on the progress of the analysis. Defaults to None
        
        Returns
        -------
        None
        """
        if len(flows) < self.w_len and con is None:
            con.put(np.array([]))
        elif len(flows) < self.w_len:
            return
        output = np.array([fs(f_0)] * len(flows))
        flows = np.array(flows).reshape(-1, self.w_len)
        flow_means = np.mean(flows, axis=1)
        flow_mean_deltas = flow_means[1:] - flow_means[:-1]
        flow_stds = np.std(flows, axis=1)
        if reporter != None:
            register_reporter(reporter)
        for i in atpbar(range(len(flow_means)-1), name="Labelling flow states"):
            if flow_stds[i] < self.f_thresh:
                w_std_i = 0.5 * self.f_thresh
            else:
                w_std_i = flow_stds[i]
            w_mean_delta = flow_mean_deltas[i]
            w_mean_i_1 = flow_means[i+1]
            curpos = (i+1) * self.w_len
            if abs(w_mean_delta) < 2 * w_std_i:
                if w_mean_i_1 > self.f_base + 2 * w_std_i:
                    # Process Peak Inspiratory Flow
                    output[curpos:curpos + self.w_len] = fs.peak_inspiratory_flow
                elif w_mean_i_1 < self.f_base - 2 * w_std_i:
                    # Process Peak Expiratory Flow
                    output[curpos:curpos + self.w_len] = fs.peak_expiratory_flow
                else:
                    # Process No Flow
                    output[curpos:curpos + self.w_len] = fs.no_flow
            elif w_mean_i_1 > self.f_base:
                if w_mean_delta > 0:
                    # Process Inspiration Initiation
                    output[curpos:curpos + self.w_len] = fs.inspiration_initiation
                else:
                    # Process Inspiration Termination
                    output[curpos:curpos + self.w_len] = fs.inspiration_termination
            else:
                if w_mean_delta < 0:
                    # Process Expiration Initiation
                    output[curpos:curpos + self.w_len] = fs.expiration_initiation
                else:
                    # Process Expiration Termination
                    output[curpos:curpos + self.w_len] = fs.expiration_termination
        if con is not None:
            con.put(output)
        else:
            self.f_labels = np.concatenate([self.f_labels, output])
        
    def process(self, pressures, flows, p_0=ps.peep, f_0=fs.no_flow):
        """
        Maps data points from pressure and flow to enumerated states
        
        Parameters
        ----------
        pressures : array like of real
            Pressure data points
        flows : array like of real
            Flow data points
        p_0 : PressureStates enum, optional
            The initial pressure state the program assumes it is in. Defaults to peep
        f_0 : FlowStates enum, optional
            The initial flow state the program assumes it is in. Defaults to no flow.
        
        Returns
        -------
        (array like of PressureStates enum, Array like of FlowStates enum)
        """
        buffer = len(pressures) % self.w_len
        
        if buffer != 0:
            self.process_pressures(pressures[:-buffer], p_0)
            self.process_flows(flows[:-buffer], f_0)
        else:
            self.process_pressures(pressures, p_0)
            self.process_flows(flows, f_0)
        self.p_labels = np.concatenate([self.p_labels, np.array([self.p_labels[-1]] * buffer)])
        self.f_labels = np.concatenate([self.f_labels, np.array([self.f_labels[-1]] * buffer)])
class PhaseLabeller:
    """ Class for segmenting and labelling breath sub-phases
    
    Attributes
    ----------
    breaths : array like of BreathVariables
        Array containing the BreathVariables after calling process
    freq : real
        Sampling rate for the record to be processed
    max_hold : integer
        Threshold in number of data points for the amount of time a breath can be in a no flow state before considered termintated
    leak_perc_thresh : real
        The proportion of leak permitted before breath is conidered physiologically implausible and to be flagged for merging in post processing
    exp_hold_len : integer
        The time in number of data points of the expiratory hold that must occur between breaths to deflag for merging in post-processing
    permit_double_cycling : boolean
        Decide whether to merge double cycles in post-processing
    
    """
    def __init__(self):
        self.configure()
    
    def configure(self, freq=100, hold_length=0.5, leak_perc_thresh=0.66, exp_hold_len=0.05, permit_double_cycling = False):
        """ 
        Sets the constants for segmentation and post-processing
        
        Parameters
        ----------
        freq : int, optional
            Sampling rate of the input data. Defaults to 100
        hold_length : real, optional
            Threshold in seconds for the amount of time a breath can be in a no flow state before considered termintated. Defaults to 0.5s
        leak_perc_thresh : real, optional
            The proportion of leak permitted before breath is conidered physiologically implausible and to be flagged for merging in post processing. Defaults to 66%
        exp_hold_len : real, optional
            The time in seconds of the expiratory hold that must occur between breaths to deflag for merging in post-processing. Defaults to 0.05s
        permit_double_cycling : boolean, optional
            Decide whether to merge double cycles in post-processing based on exp_hold_len. Defaults to false.
            
        Returns
        -------
        None
        """
        self.breaths = []
        self.freq = freq
        self.max_hold = math.ceil(freq * hold_length)
        self.leak_perc_thresh = leak_perc_thresh
        self.exp_hold_len = math.ceil(freq * exp_hold_len)
        self.permit_double_cycling = permit_double_cycling
    
    def process(self, p_labels, f_labels, pressures, flows, post_processing=True):
        """
        Given the pressure and flow data points and labels, segments the data into breaths, identifies respiratory sub-phases, and calculates some physiological values
        
        Parameters
        ----------
        p_labels : array like of PressureStates enum
            The PressureStates labels generated from StateMapper
        f_labels : array like of FlowStates enum
            The FlowStates labels generated from StateMapper
        pressures : array like of real
            Pressure data points
        flows : array like of real
            Flow data points
        post_processing : boolean, optional
            Flag for deciding whether to run post processing or not. Defaults to True
            
        Returns
        -------
        None
        """
        if type(pressures) is not np.array:
            pressures = np.array(pressures)
        if type(flows) is not np.array:
            flows = np.array(flows)
        print("Segmenting into breaths")
        self.breaths += [BreathVariables()]
        self.breaths[-1].breath_end = 0
        while(self.breaths[-1].breath_end != len(f_labels)):
            self.breaths += [self.__get_next_breath(f_labels, self.breaths[-1].breath_end)]
        # First and last breaths are usually inaccurate
        if len(self.breaths) > 1:
            self.breaths = self.breaths[1:]
            print(str(len(self.breaths)) + " breaths identified")
            for i in atpbar(range(len(self.breaths)), name="Processing breaths"):
                self.breaths[i].breath_number = i+1
                self.__information_approach(p_labels, f_labels, self.breaths[i])
                self.__calculate_features(self.breaths[i], pressures, flows)
            if post_processing:
                self.__post_process(p_labels, f_labels, pressures, flows)
        else:
            self.breaths = []
            print("Warning: No breaths identified")
    
    def get_breaths(self, length_units="ms"):
        """
        Returns the segmented breaths and calculated features as a pandas dataframe. See BreathVariables for list of variables returned
        
        Parameters
        ----------
        length_units : string, optional
            Unit to use for length calculations, accepts 'ms' and 's' for milliseconds and seconds respectively. Defaults to ms
        
        Returns
        --------
        Pandas Dataframe
            Table of segmented breaths and charactersitics for each breath with lengths scaled according to given unit
        """
        df = pd.DataFrame([vars(x) for x in self.breaths])
        if length_units == "ms":
            df[list(filter(lambda x : "length" in x, df.columns))] *= 1000 / self.freq
        elif length_units == "s":
            df[list(filter(lambda x : "length" in x, df.columns))] *= 1 / self.freq
        return df[["breath_number", "breath_start", "breath_end", "inspiration_initiation_start", "peak_inspiratory_flow_start",
                  "inspiration_termination_start", "inspiratory_hold_start", "expiration_initiation_start",	"peak_expiratory_flow_start",
                  "expiration_termination_start", "expiratory_hold_start", "pressure_rise_start", "pip_start", "pressure_drop_start",
                  "peep_start", "inspiration_initiation_length", "peak_inspiratory_flow_length",
                  "inspiration_termination_length", "inspiratory_hold_length", "expiration_initiation_length", "peak_expiratory_flow_length",
                  "expiration_termination_length", "expiratory_hold_length", "pressure_rise_length", "pip_length", "pressure_drop_length",
                  "peep_length", "pip_to_no_flow_length", "peep_to_no_flow_length", "lung_inflation_length", "total_inspiratory_length",
                  "lung_deflation_length", "total_expiratory_length", "inspiratory_volume", "expiratory_volume", "max_inspiratory_flow",
                  "max_expiratory_flow", "max_pressure", "min_pressure", "pressure_flow_correlation"]]
        
    def get_breaths_raw(self):
        """
        Returns the segmented breaths and calculated features as a pandas dataframe. See BreathVariables for list of variables returned
        
        Returns
        --------
        Pandas Dataframe
            Table of segmented breaths and charactersitics for each breath
        """
        return pd.DataFrame([vars(x) for x in self.breaths])
    
    def get_breath_annotations(self, N, p_states=list(ps), f_states=list(fs)):
        """ 
        Returns a Nx3 dataframe containing key points of breaths mapped to indices to be used with GUI annotator for viewing
        
        Parameters
        ----------
        N : int
            Length of the sample that was analyzed (in terms of data points)
        p_states : array like of PressureStates, optional
            The pressure states from each breath that you would like mapped. Defaults to all enums in PressureStates.
        f_states : array like of FlowStates, optional
            The flow states from each breath that you woudld like mapped. Defaults to all enums in FlowStates
        
        Returns
        -------
        Pandas Dataframe
            Dataframe containing keypoints at each index of the data on which the analysis was performed
        """
        output = np.full((N,3), -1)
        output[:,0] = np.arange(N)
        breaths = self.get_breaths_raw()
        for p in p_states:
            if p == ps.pressure_rise:
                output[breaths["pressure_rise_start"]-1,1] = ps.pressure_rise.value
            elif p == ps.pip:
                output[breaths["pip_start"]-1,1] = ps.pip.value
            elif p == ps.pressure_drop:
                output[breaths["pressure_drop_start"]-1,1] = ps.pressure_drop.value
            elif p == ps.peep:
                output[breaths["peep_start"]-1,1] = ps.peep.value
        for f in f_states:
            if f == fs.inspiration_initiation:
                output[breaths["inspiration_initiation_start"]-1,2] = fs.inspiration_initiation.value
            elif f == fs.peak_inspiratory_flow:
                output[breaths["peak_inspiratory_flow_start"]-1,2] = fs.peak_inspiratory_flow.value
            elif f == fs.inspiration_termination:
                output[breaths["inspiration_termination_start"]-1,2] = fs.inspiration_termination.value
            elif f == fs.no_flow:
                output[breaths["inspiratory_hold_start"]-1,2] = fs.no_flow.value
                output[breaths["expiratory_hold_start"]-1,2] = fs.no_flow.value
            elif f == fs.expiration_initiation:
                output[breaths["expiration_initiation_start"]-1,2] = fs.expiration_initiation.value
            elif f == fs.peak_expiratory_flow:
                output[breaths["peak_expiratory_flow_start"]-1,2] = fs.peak_expiratory_flow.value
            elif f == fs.expiration_termination:
                output[breaths["expiration_termination_start"]-1,2] = fs.expiration_termination.value
        output = output[output[:,1:].sum(axis=1) != -2,:]
        output = pd.DataFrame(output)
        output.columns = ["index","pressure_annotations","flow_annotations"]
        return output
    
    def __get_next_breath(self, labels, start):
        """
        Identifies the next breath in the record based on Inspiration-Inspiration interval
        
        Parameters
        ----------
        labels : array like of FlowStates enum
            The array of flow labels calculated from a StateMapper object
        start : integer
            Index from which to start searching for a breath
        
        Returns
        -------
        BreathVariables object
            A breath object containing the start and end points
        """
        running_hold = 0
        expiration_encountered = False
        for i in range(start, len(labels)):
            if labels[i] is fs.inspiration_initiation or labels[i] is fs.peak_inspiratory_flow:
                running_hold = 0
                # If this is the start of the recording, the start index may be inaccurate so we reset it here
                if start == 0:
                    start = i
                    expiration_encountered=False
                if expiration_encountered:
                    breath = BreathVariables()
                    breath.breath_start = start
                    breath.breath_end = i
                    return breath
            elif labels[i] is fs.expiration_initiation or labels[i] is fs.peak_expiratory_flow or labels[i] is fs.expiration_termination or running_hold > self.max_hold:
                expiration_encountered = True
            elif labels[i] is fs.no_flow:
                running_hold += 1
                
        # If code reaches this point then it is the last breath of the record
        breath = BreathVariables()
        breath.breath_start = start
        breath.breath_end = len(labels)
        return breath
    
    def __maximise_information_gain(self, labels, target_classes):
        """
        Finds the split on the given labels which maximises information gain
        
        Parameters
        ----------
        labels : array like of PressureStates or FlowStates
            An array of labels (enumerated flow/pressure states)
        target_classes : array like of PressureStates or FlowStates
            An array of labels (enumerated flow/pressure states) to use to calculate information gain
        
        Returns
        -------
        (int, array like of PressureStates or FlowStates, array like of PressureStates or FlowStates)
            Returns the index of the split, the states up to index, states from index to the end
        """
        some_exists = False
        for target_class in target_classes:
            if target_class in labels:
                some_exists = True
                break
        if not some_exists:
            return (0, np.array([]), labels)
        if len(labels) == 0:
            return (0, np.array([]), labels)
        elif len(labels) == 1:
            for target_class in target_classes:
                if target_class in labels:
                    return(1, labels, np.array([]))
            return (0, np.array([]), labels)
        # Find p
        xlen = len(labels)
        forward = np.arange(1,xlen)
        backward = np.arange(xlen-1,0,step=-1)
        p = 0
        p2 = 0
        for target_class in target_classes:
            p += (labels == target_class).cumsum()[:-1]
        p2 = (p[-1] - np.copy(p)) / backward
        p = p / forward
        inf = ((-p * np.log(p + 1E-7)) * forward + (-p2 * np.log(p2 + 1E-7)) * backward) / xlen
        p_prime = 1-p
        p2_prime = 1-p2
        inf += ((-p_prime * np.log(p_prime + 1E-7 )) * forward + (-p2_prime * np.log(p2_prime + 1E-7)) * backward) / xlen
        idx = np.argmin(inf) + 1
        return (idx, labels[:idx], labels[idx:])
       
    def __information_approach(self, p_labels, f_labels, breath):
        """
        Tries to identify sub-phases of each breath based on maximising information gain on splitting
        
        Parameters
        ----------
        p_labels : array like of PressureStates enum
            Pressure labels for the record calculated using StateMapper
        f_labels : array like of FlowStates enum
            Flow labels for the record calculated using StateMapper
        breath : BreathVariables object
            BreathVariables object for the breath to calculate sub phases
        
        Returns
        -------
        None
        """
        p_labels = p_labels[breath.breath_start:breath.breath_end]
        f_labels = f_labels[breath.breath_start:breath.breath_end]
        labels = f_labels
        # Inspiration initiation at breath start by segmentation definition
        breath.inspiration_initiation_start = breath.breath_start
        # Find Peak inspiratory flow start by finding end of split
        breath.peak_inspiratory_flow_start, _, labels = self.__maximise_information_gain(labels, [fs.inspiration_initiation])
        breath.peak_inspiratory_flow_start += breath.breath_start
        # Find Inspiration termination start by finding end of split
        breath.inspiration_termination_start, _, labels = self.__maximise_information_gain(labels, [fs.peak_inspiratory_flow])
        breath.inspiration_termination_start += breath.peak_inspiratory_flow_start
        # Find Inspiratory Hold start by finding end of split
        breath.inspiratory_hold_start, _, labels = self.__maximise_information_gain(labels, [fs.inspiration_termination])
        breath.inspiratory_hold_start += breath.inspiration_termination_start
        # Find Peak expiratory flow start by finding end of split
        breath.peak_expiratory_flow_start, _, labels = self.__maximise_information_gain(labels, [fs.expiration_initiation])
        breath.peak_expiratory_flow_start += breath.inspiratory_hold_start
        # Find Expiratory hold start by finding end of split
        no_flow, _, labels = self.__maximise_information_gain(labels, [fs.no_flow])
        if no_flow == 0:
            breath.expiratory_hold_start = breath.breath_end
        else:
            breath.expiratory_hold_start = breath.peak_expiratory_flow_start + no_flow        
        # Find Expiration Termination Start by finding end of split
        templabels = f_labels[breath.peak_expiratory_flow_start - breath.breath_start : breath.expiratory_hold_start - breath.breath_start]
        breath.expiration_termination_start, _, labels = self.__maximise_information_gain(templabels, [fs.peak_expiratory_flow])
        breath.expiration_termination_start += breath.peak_expiratory_flow_start
        # Find expiration initiation start by finding end of split
        templabels = f_labels[breath.inspiratory_hold_start - breath.breath_start : breath.peak_expiratory_flow_start - breath.breath_start]
        breath.expiration_initiation_start, _, labels = self.__maximise_information_gain(templabels, [fs.no_flow])
        breath.expiration_initiation_start += breath.inspiratory_hold_start
        
        labels = p_labels
        # Find pip start by finding end of split
        breath.pip_start, _, labels = self.__maximise_information_gain(labels, [ps.pressure_rise])
        breath.pip_start += breath.breath_start
        
        # Find pressure drop start by finding end of split
        breath.pressure_drop_start, _, labels = self.__maximise_information_gain(labels, [ps.pip])
        breath.pressure_drop_start += breath.pip_start
        
        # Find peep start by finding start of split
        breath.peep_start, _, labels = self.__maximise_information_gain(labels, [ps.pressure_drop])
        breath.peep_start += breath.pressure_drop_start
        
        # Find pressure rise start by finding start of split
        breath.pressure_rise_start, _, labels = self.__maximise_information_gain(p_labels[:breath.pip_start - breath.breath_start], [ps.peep])
        breath.pressure_rise_start += breath.breath_start
    
    def __calculate_features(self, breath, pressures, flows):
        """
        Calculates the values relevant for physiology like tidal volumes and respiratory phase lengths
        
        Parameters
        ----------
        breath : BreathVariables object
            The breath for which to calculate the physiological values
        pressures : array like of real
            Pressure data points
        flows : array like of real
            Flow data points
        
        Returns
        -------
        None
        """
        p = np.array(pressures[breath.breath_start:breath.breath_end])
        f = np.array(flows[breath.breath_start:breath.breath_end])

        # Pressure phases
        breath.pressure_rise_length = breath.pip_start - breath.pressure_rise_start
        breath.pip_length = breath.pressure_drop_start - breath.pip_start
        breath.pressure_drop_length = breath.peep_start - breath.pressure_drop_start
        breath.peep_length = breath.breath_end - breath.peep_start
        
        # Flow phases
        breath.inspiration_initiation_length = breath.peak_inspiratory_flow_start - breath.inspiration_initiation_start
        breath.peak_inspiratory_flow_length = breath.inspiration_termination_start - breath.peak_inspiratory_flow_start
        breath.inspiration_termination_length = breath.inspiratory_hold_start - breath.inspiration_termination_start
        breath.inspiratory_hold_length = breath.expiration_initiation_start - breath.inspiratory_hold_start
        breath.expiration_initiation_length = breath.peak_expiratory_flow_start - breath.expiration_initiation_start
        breath.peak_expiratory_flow_length = breath.expiration_termination_start - breath.peak_expiratory_flow_start
        breath.expiration_termination_length = breath.expiratory_hold_start - breath.expiration_termination_start
        breath.expiratory_hold_length = breath.breath_end - breath.expiratory_hold_start
        breath.lung_inflation_length = breath.inspiratory_hold_start - breath.inspiration_initiation_start
        breath.total_inspiratory_length = breath.expiration_initiation_start - breath.inspiration_initiation_start
        breath.lung_deflation_length = breath.expiratory_hold_start - breath.expiration_initiation_start
        breath.total_expiratory_length = breath.breath_end - breath.expiration_initiation_start
        breath.pip_to_no_flow_length = breath.inspiratory_hold_start - breath.pip_start
        breath.peep_to_no_flow_length = breath.expiratory_hold_start - breath.peep_start
        
        # Volumes
        breath.inspiratory_volume = f[f > 0].sum()
        breath.expiratory_volume = np.abs(f[f < 0].sum())
        breath.max_inspiratory_flow = f.max()
        breath.max_expiratory_flow = f.min()
        
        # Pressures
        breath.max_pressure = p.max()
        breath.min_pressure = p.min()
        
        # Correlation
        breath.pressure_flow_correlation = np.corrcoef(p,f)[0,1]
    
    def __post_process(self, p_labels, f_labels, pressures, flows):
        """ 
        Performs merging of adjacent breaths dependent on whether inspiration and expiration volumes match
        
        Parameters
        ----------
        p_labels : array like of PressureStates enum
            Pressure labels for the record calculated using StateMapper
        f_labels : array like of FlowStates enum
            Flow labels for the record calculated using StateMapper
        pressures : array like of real
            Pressure data points
        flows : array like of real
            Flow data points
       
        Returns
        -------
        None
        """
        merged_breaths = [self.breaths[0]]
        begin_merge = False
        insp_sum = 0
        exp_sum = 0
        error_start = 0
        for i in atpbar(range(1,len(self.breaths)), name="Post-processing"):
            if not begin_merge:
                breath_leak_perc = (self.breaths[i].inspiratory_volume - self.breaths[i].expiratory_volume) / self.breaths[i].inspiratory_volume
                if abs(breath_leak_perc) > self.leak_perc_thresh:
                    if breath_leak_perc < 0 and self.breaths[i-1].expiratory_hold_length <= self.exp_hold_len:
                        error_start = i - 1
                        merged_breaths.pop()
                        begin_merge = True
                        insp_sum += self.breaths[i-1].inspiratory_volume + self.breaths[i].inspiratory_volume
                        exp_sum += self.breaths[i-1].expiratory_volume + self.breaths[i].expiratory_volume
                    elif breath_leak_perc > 0 and self.breaths[i-1].expiratory_hold_length <= self.exp_hold_len:
                        begin_merge = True
                        error_start = i
                        insp_sum += self.breaths[i].inspiratory_volume
                        exp_sum += self.breaths[i].expiratory_volume
                    else:
                        merged_breaths += [self.breaths[i]]
                else:
                    merged_breaths += [self.breaths[i]]
            else:
                if ((abs(insp_sum - exp_sum)/insp_sum < self.leak_perc_thresh or self.breaths[i-1].expiratory_hold_length > self.exp_hold_len) and error_start != i-1) or (self.breaths[i].pressure_flow_correlation > 0.2 and not self.permit_double_cycling):
                    # Begin to merge breaths
                    begin_merge = False
                    insp_sum = 0
                    exp_sum = 0
                    merged_breath = BreathVariables()
                    merged_breath.breath_start = self.breaths[error_start].breath_start
                    merged_breath.breath_end = self.breaths[i-1].breath_end
                    self.__information_approach(p_labels, f_labels, merged_breath)
                    self.__calculate_features(merged_breath, pressures, flows)
                    merged_breaths += [merged_breath]
                    # Check if current breath needs to be merged
                    breath_leak_perc = (self.breaths[i].inspiratory_volume - self.breaths[i].expiratory_volume) / self.breaths[i].inspiratory_volume
                    if abs(breath_leak_perc) > self.leak_perc_thresh:
                        if breath_leak_perc < 0 and self.breaths[i].expiratory_hold_length <= self.exp_hold_len:
                            merged_breaths.pop()
                            begin_merge = True
                            insp_sum += self.breaths[i-1].inspiratory_volume + self.breaths[i].inspiratory_volume
                            exp_sum += self.breaths[i-1].expiratory_volume + self.breaths[i].expiratory_volume
                        elif breath_leak_perc > 0 and self.breaths[i].expiratory_hold_length <= self.exp_hold_len:
                            begin_merge = True
                            error_start = i
                            insp_sum += self.breaths[i].inspiratory_volume
                            exp_sum += self.breaths[i].expiratory_volume
                        else:
                            merged_breaths += [self.breaths[i]]
                    else:
                        merged_breaths += [self.breaths[i]]
                else:
                    insp_sum += self.breaths[i].inspiratory_volume
                    exp_sum += self.breaths[i].expiratory_volume
        
        self.breaths = merged_breaths
        for i in atpbar(range(len(self.breaths)), name="Re-numbering breaths"):
            self.breaths[i].breath_number = i+1


def pre_process_ventilation_data(path, cols):
    """
    Loads ventilation data and fills in NA values via linear interpolation
    
    Parameters
    ----------
    path : string
        Path to the input data file
    cols : array like of integers
        Integers corresponding to the columns that are to be used for analysis. Assumes the columns refer to an index/time, pressure, and flows column.
    
    Returns
    -------
    Pandas Dataframe
    """
    data = pd.read_csv(path, usecols=cols)
    # Impute missing values for pressure, flow, volume
    for i in range(0,data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i].interpolate()
    return data

def correct_baseline(data, window):
    """
    Attempts to do basic window mean centering of the data to correct baseline wander
    
    Parameters
    ----------
    data : Pandas Dataframe
        The data obtained from pre_process_ventilation_data call
    window : int
        The size of the window to use for baseline correction
    """
    for i in range(0,int(len(data)/window)):
        if (i+1)*window < data.shape[0]:
            data[(i*window):((i+1)*window)] = data[(i*window):((i+1)*window)] - np.mean(data[(i*window):((i+1)*window)])
        else:
            data[(i*window):] = data[(i*window):] - np.mean(data[(i*window):])
            


class BreathVariables:
    """
    Structure to hold measurements of each breath.
    
    Attributes
    ----------
    breath_number : int
        Index for the breath with respect to the analysis that was run
    breath_start : int
        The index in the input waveform data corresponding to the start of the current breath
    breath_end : int
        The index in the input waveform data corresponding to the end of the current breath
    pressure_rise_start : int
        The index in the input waveform data corresponding to the start of the pressure rise phase of the current breath
    pip_start : int
        The index in the input waveform data corresponding to the start of the peak inflation pressure phase of the current breath
    pressure_drop_start : int
        The index in the input waveform data corresponding to the start of the pressure drop phase of the current breath
    peep_start : int
        The index in the input waveform data corresponding to the start of the positive end expiratory pressure phase of the current breath
    inspiration_initiation_start : int
        The index in the input waveform data corresponding to the start of the inspiration initiation phase of the current breath
    inspiratory_hold_start : int
        The index in the input waveform data corresponding to the start of the inspiratory hold phase of the current breath
    peak_inspiratory_flow_start : int
        The index in the input waveform data corresponding to the start of the peak inspiratory flow phase of the current breath
    inspiration_termination_start : int
        The index in the input waveform data corresponding to the start of the inspiration termination phase of the current breath  
    inspiratory_hold_start : int
        The index in the input waveform data corresponding to the start of the inspiratory hold phase of the current breath
    expiration_initiation_start : int
        The index in the input waveform data corresponding to the start of the expiration initiation phase of the current breath
    peak_expiratory_flow_start : int
        The index in the input waveform data corresponding to the start of the peak expiratory flow phase of the current breath
    expiration_termination_start : int
        The index in the input waveform data corresponding to the start of the expiration termination phase of the current breath
    expiratory_hold_start : int
        The index in the input waveform data corresponding to the start of the expiratory hold phase of the current breath
    pressure_rise_length : int
        The length of the pressure rise phase of the current breath in terms of number of time units
    pip_length : int
        The length of the peak inspiratory pressure phase of the current breath in terms of number of time units
    peep_length : int
        The length of the positive end expiratory pressure phase of the current breath in terms of number of time units
    inspiration_initiation_length : int
        The length of the inspiration initiation phase of the current breath in terms of number of time units
    peak_inpiratory_flow_length : int
        The length of the peak inspiratory flow phase of the current breath in terms of number of time units
    inspiration_termination_length : int
        The length of the inspiration termination phase of the current breath in terms of number of time units
    inspiratory_hold_length : int
        The length of the inspiratory hold phase of the current breath in terms of number of time units
    expiration_initiation_length : int
        The length of the expiration initiation phase of the current breath in terms of number of time units
    peak_expiratory_flow_length : int
        The length of the peak expiratory flow phase of the current breath in terms of number of time units
    expiration_termination_length : int
        The length of the expiration termination phase of the current breath in terms of number of time units
    expiratory_hold_length : int
        The length of the expiratory hold phase of the current breath in terms of number of time units
    pip_to_no_flow_length : int
        The length of the period from the start of peak inspiratory pressure phase to the start of inspiratory hold phase in terms of number of time units
    peep_to_no_flow : int
        The length of the period from the start of the positive end expiratory pressure phase to the start of expiratory hold phase in terms of the number of time units
    lung_inflation_length : int
        The length of the period from the start of inspiration initiation phase to the start of the inspiratory hold phase in terms of number of time units
    lung_deflation_length : int
        The length of the period from the start of expiration initiation phase to the start of the expiratory hold phase in terms of number of time units
    total_inspiratory_length : int
        The length of the period from the start of inspiration initiation phase to the start of the expiration initiation in terms of number of time units
    total_expiratory_length : int
        The length of the period from the start of expiratory initiation phase to the end of the breath in terms of number of time units
    inspiratory_volume : real
        The signed volume of inspiration calculated by summing the positive flow values in the current breath
    expiratory_volume : real
        The signed volume of expiration calculated by summing the negative flow values in the current breath
    max_inspiratory_flow : real
        The most positive flow value in the current breath
    max_expiratory_flow : real
        The most negative flow value in the current breath
    max_pressure : real
        The largest pressure value in the current breath
    min_pressure : real
        The smallest pressure value in the current breath
    pressure_flow_correlation : real
        Pearson correlation coefficient of the pressure and flow for the current breath
    
    Methods
    --------
    valid()
        Checks if the indices of the key points are physiologically valid in terms of order
    """
    def __init__(self):
        # Points in time
        self.breath_number = None
        self.breath_start = None
        self.breath_end = None
        self.pressure_rise_start = None
        self.pip_start = None
        self.pressure_drop_start = None
        self.peep_start = None
        self.inspiration_initiation_start = None
        self.peak_inspiratory_flow_start = None
        self.inspiration_termination_start = None
        self.inspiratory_hold_start = None
        self.expiration_initiation_start = None
        self.peak_expiratory_flow_start = None
        self.expiration_termination_start = None
        self.expiratory_hold_start = None
        # Length of phases
        self.pressure_rise_length = None
        self.pip_length = None
        self.pressure_drop_length = None
        self.peep_length = None
        self.inspiration_initiation_length = None
        self.peak_inspiratory_flow_length = None
        self.inspiration_termination_length = None
        self.inspiratory_hold_length = None
        self.expiration_initiation_length = None
        self.peak_expiratory_flow_length = None
        self.expiration_termination_length = None
        self.expiratory_hold_length = None
        self.pip_to_no_flow_length = None
        self.peep_to_no_flow_length = None
        self.lung_inflation_length = None
        self.total_inspiratory_length = None
        self.lung_deflation_length = None
        self.total_expiratory_length = None
        # Volumes
        self.inspiratory_volume = None
        self.expiratory_volume = None
        # Extreme values
        self.max_inspiratory_flow = None
        self.max_expiratory_flow = None
        self.max_pressure = None
        self.min_pressure = None
        # Misc
        self.pressure_flow_correlation = None
    
    def valid(self):
        """
        Checks if the indices of the key points are physiologically valid in terms of order
        
        Returns
        -------
        boolean
            Indicates if the indices of the key points are physiologically valid in terms of order
        """
        return (self.pressure_rise_start <= self.pip_start <= self.pressure_drop_start <= self.peep_start) and ((self.pressure_rise_length + self.pip_length + self.pressure_drop_start + self.peep_length) == (self.breath_end - self.breath_start)) and (self.inspiration_initiation_start <= self.peak_inspiratory_flow_start <= self.inspiration_termination_start <= self.inspiratory_hold_start <= self.expiration_initiation_start <= self.peak_expiratory_flow_start <= self.expiration_termination_start <= self.expiratory_hold_start) and ((self.inspiration_initiation_length + self.peak_inspiratory_flow_length + self.inspiration_termination_length + self.inspiratory_hold_length + self.expiration_initiation_length + self.peak_expiratory_flow_length + self.expiration_termination_length + self.expiratory_hold_length) == (self.breath_end - self.breath_start))
