import scipy
import numpy as np
from sklearn.preprocessing import StandardScaler
from cycler import cycler
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import statsmodels.api as sm
from scipy.fft import rfft, rfftfreq
import simdkalman

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos
    
def PCA_decomposition(A):
  M = np.mean(A, axis=0)
  C = A - M
  V = np.cov(C)
  values, vectors = np.linalg.eig(V)
    
  return values, vectors

def X_to_TS(X_i):
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

def PCA_timeseries(signal):
    ##PCA for time series after https://www.kaggle.com/code/leokaka/pca-for-time-series-analysis

    N = len(signal)
    L = 20 # The window length
    K = N - L + 1  # number of columns in the trajectory matrix
    t = np.arange(0,N)
    X = np.column_stack([signal[i:i+L] for i in range(0,K)])
    d = 15
    sX = StandardScaler(copy=True)
    X_trans = sX.fit_transform(X)
    values, vectors = PCA_decomposition(X)
    U_k = vectors
    X_elem_pca = np.array([np.dot(np.dot(np.expand_dims(U_k[:,i], axis=1), np.expand_dims(U_k[:,i].T, axis=0)), X) for i in range(0,d)])
    n = min(70,d) # In case of noiseless time series with d < 12.
    F_trend = X_to_TS(X_elem_pca[[0,5]].sum(axis=0))
    return F_trend

def import_ventilator_rate(filename, timestart, timeend):
    ventilator_timestamps=[]
    ventilator_rate=[]
    skip_line=True
    with open(filename) as f: 
        for line in f:
            if skip_line:
                skip_line=False
            else:
                currentline = line.split(",")
                time=float(currentline[0])
                time=time/1000
                if currentline[31]!='':
                    rate=float(currentline[31])
                    if time>timestart and time<timeend:
                        ventilator_timestamps.append(time)
                        ventilator_rate.append(rate)
    return ventilator_timestamps, ventilator_rate

def import_impedance_rate(filename, timestart, timeend):
    impedance_timestamps=[]
    impedance_rate=[]
    with open(filename) as f: 
        for line in f:
            currentline = line.split(",")
            time=float(currentline[0])
            rate=int(currentline[1][:-1])
            if time>timestart and time<timeend:
                impedance_timestamps.append(time)
                impedance_rate.append(rate)
    return impedance_timestamps, impedance_rate

def CHROM_signal(red, green, blue):
    red=np.array(red)
    green=np.array(green)
    blue=np.array(blue)
    mean_red=np.mean(red)
    mean_blue=np.mean(blue)
    mean_green=np.mean(green)
    
    R_s=0.7682/mean_red*red
    B_s=0.5121/mean_blue*blue
    G_s=0.3841/mean_green*green
    X=(R_s-G_s)/(0.7672-0.5121)
    Y=(R_s+G_s-2*B_s)/(0.7682+0.5121-0.7682)
    
    butter= butter_bandpass(0.5, 5, 30, order=7)
    X_f=scipy.signal.sosfiltfilt(butter, X)
    Y_f=scipy.signal.sosfiltfilt(butter, Y)
    alpha=np.std(X_f)/np.std(Y_f)
    
    R_f=scipy.signal.sosfiltfilt(butter, red)
    G_f=scipy.signal.sosfiltfilt(butter, green)
    B_f=scipy.signal.sosfiltfilt(butter, blue)
    heart_rate_signal=3*(1-alpha/2)*R_f-2*(1+alpha/2)*G_f+3*alpha/2*B_f.T
    return heart_rate_signal

def import_ECG_rate(filename, timestart, timeend):
    ECG_timestamps=[]
    ECG_rate=[]
    with open(filename) as f: 
        for line in f:
            currentline = line.split(",")
            time=float(currentline[0])
            rate=int(currentline[1][:-1])
            if time>timestart and time<timeend and rate>0:
                ECG_timestamps.append(time)
                ECG_rate.append(rate)
    return ECG_timestamps, ECG_rate

    
def import_ventilator_volumes(filename, timestart, timeend):
    ventilator_volume_timestamps=[]
    ventilator_tidal_volume=[]
    skip_line=True
    with open(filename) as f: 
        for line in f:
            if skip_line:
                skip_line=False
            else:
                currentline = line.split(",")
                time=float(currentline[0])
                time=time/1000
                if currentline[25]!='':
                    rate=float(currentline[25])
                    if time>timestart and time<timeend:
                        ventilator_volume_timestamps.append(time)
                        ventilator_tidal_volume.append(rate)
    return ventilator_volume_timestamps, ventilator_tidal_volume

def import_spO2(filename, timestart, timeend):
    pulse_oxymeter_timestamps=[]
    pulse_oxymeter_spO2=[]
    with open(filename) as f: 
        for line in f:
            currentline = line.split(",")
            time=float(currentline[0])
            rate=int(currentline[1][:-1])
            if time>timestart and time<timeend:
                pulse_oxymeter_timestamps.append(time)
                pulse_oxymeter_spO2.append(rate)
    return pulse_oxymeter_timestamps, pulse_oxymeter_spO2

def skin_pixel_identification(image):
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    masked_img=cv2.bitwise_and(image,image, mask=global_mask)
    return masked_img


def import_ventilator_flow(file, timestart, timeend):
    ventilator_flow=[]
    ventilator_time=[]
    ventilator_pressure=[]
    with open(file) as f: 
        for line in f:
            currentline = line.split(",") 
            if  currentline[0]!='Time [ms]':
                        time=float(currentline[0])
                        time=time/1000
                        if currentline[5]!='' and currentline[4]!='' and time>timestart and time<timeend:
                                    ventilator_flow.append(float(currentline[5]))
                                    ventilator_time.append(float(currentline[0]))
                                    ventilator_pressure.append(float(currentline[4]))
    return ventilator_time, ventilator_flow, ventilator_pressure

def coverage_probability(truth, test, kappa):
    truth=np.array(truth)
    truth_upper=truth+kappa/100*np.mean(truth)
    truth_lower=truth-kappa/100*np.mean(truth)
    count_in=0
    for i in range(len(truth)):
        if test[i]>truth_lower[i] and test[i]<truth_upper[i]:
            count_in+=1
    
    tdi=count_in/len(truth)
    return tdi

def concordance_correlation_coefficient(array1, array2):
    array1=np.array(array1)
    array2=np.array(array2)
    mu_1=np.mean(array1)
    mu_2=np.mean(array2)
    std_1=np.std(array1)
    std_2=np.std(array2)
    std_12=np.sum((array1 - array1.mean())*(array2 - array2.mean()))/array1.shape[0]
    ccc=2*std_12/(std_1**2+std_2**2+(mu_1-mu_2)**2)
    return ccc

def mean_square_diff(array1, array2):
    square_diff=[]
    for i in range(len(array1)):
        square_diff.append((array1[i]-array2[i])**2)
    mean_diff=np.mean(np.array(square_diff))
    return mean_diff

def mean_absolute_diff(array1, array2):
    square_diff=[]
    for i in range(len(array1)):
        square_diff.append(abs(array1[i]-array2[i]))
    mean_diff=np.mean(np.array(square_diff))
    return mean_diff

    
def statistical_analysis(ground_truth_signal, reference_signal, ground_truth_timestamps, reference_timestamps, kappa_1, kappa_2):
    ground_truth_signal=np.array(ground_truth_signal)
    reference_signal=np.array(reference_signal)
    ground_truth_timestamps=np.array(ground_truth_timestamps)
    reference_timestamps=np.array(reference_timestamps)
    matched_ground_truth=[]
    matched_reference=[]
    if len(ground_truth_signal)>=len(reference_signal):
        for i in range(len(reference_signal)-1):
            timestart=reference_timestamps[i]
            timeend=reference_timestamps[i+1]
            value_indices_in_window=np.where(np.logical_and(ground_truth_timestamps>= timestart, ground_truth_timestamps<= timeend))[0]
            
            values_in_window=[]
            if len(value_indices_in_window>0):
                for j in value_indices_in_window:
                    values_in_window.append(ground_truth_signal[j])
                
                matched_ground_truth.append(np.mean(np.array(values_in_window)))
                matched_reference.append(reference_signal[i])
            
    else:
        for i in range(len(ground_truth_signal)-1):
            timestart=ground_truth_timestamps[i]
            timeend=ground_truth_timestamps[i+1]
            value_indices_in_window=np.where(np.logical_and(reference_timestamps>= timestart, reference_timestamps<= timeend))[0]
            
            values_in_window=[]
            if len(value_indices_in_window)>0:
                for j in value_indices_in_window:
                    values_in_window.append(reference_signal[j])
                matched_reference.append(np.mean(np.array(values_in_window)))
                matched_ground_truth.append(ground_truth_signal[i])
        

    print("Data points:",len(matched_ground_truth))
        
    mean_diff = mean_absolute_diff(matched_ground_truth, matched_reference)
    print("MAD:", mean_diff)

    mean_square_error = mean_square_diff(matched_ground_truth, matched_reference)
    print("MSE:", mean_square_error)

    ccc=concordance_correlation_coefficient(matched_ground_truth, matched_reference)
    print("CCC:",ccc)

    cp=coverage_probability(matched_ground_truth, matched_reference, kappa_1)
    print("CP", kappa_1,  "%:",cp)

    cp=coverage_probability(matched_ground_truth, matched_reference, kappa_2)
    print("CP", kappa_2,"%:",cp)

    #f, ax = plt.subplots(1, figsize = (8,5))
    #sm.graphics.mean_diff_plot(np.array(matched_ground_truth), np.array(matched_reference), ax = ax)
    #plt.show()

  
def find_ac_and_dc_signal(filtered_signal, intervall):
    signal_ac_signal=[]
    signal_dc_signal=[]
    signal_peaks_values=[]
    
    peaks_signal,_ = find_peaks(filtered_signal, width=3, distance=6)
    for i in range(len(peaks_signal)):
        signal_peaks_values.append(filtered_signal[peaks_signal[i]])

    sorted_signal_values=np.sort(signal_peaks_values)
    signal_length=np.size(sorted_signal_values)
    mean_signal=np.mean(sorted_signal_values[int(signal_length/10):int(signal_length*9/10)])
    std_signal=np.std(sorted_signal_values[int(signal_length/10):int(signal_length*9/10)])
    
    
    for time in range(int((len(filtered_signal)-intervall)/30)):
        result_signal = np.where(np.logical_and(peaks_signal>= 30*time, peaks_signal<= intervall+30*time))[0]
        ac_signal, dc_signal, valid_peaks=0,0,0
        for peak in range(len(result_signal)-1):
            if filtered_signal[peaks_signal[result_signal[peak]]]>mean_signal-3*std_signal and filtered_signal[peaks_signal[result_signal[peak]]]<mean_signal+3*std_signal:
                heart_beat_start, heart_beat_end=peaks_signal[result_signal[peak]], peaks_signal[result_signal[peak+1]]
                heart_beat_signal=filtered_signal[heart_beat_start:heart_beat_end]
                
                ac_signal+=heart_beat_signal[0]-np.min(heart_beat_signal)
                dc_signal+=np.mean(heart_beat_signal)
                valid_peaks+=1
        signal_ac_signal.append((ac_signal)/valid_peaks)
        signal_dc_signal.append(dc_signal/valid_peaks)
    
    return signal_ac_signal, signal_dc_signal
    


def infrared_oxygenation(ts1, filtered_red, filtered_ir, intervall):
    red_ac_signal,red_dc_signal=find_ac_and_dc_signal(filtered_red, intervall)
    ir_ac_signal,ir_dc_signal=find_ac_and_dc_signal(filtered_ir, intervall)
    time_spO2=np.linspace(ts1[intervall], ts1[-1],num=int((len((filtered_red))-intervall)/30))

    red_signal=np.divide(red_ac_signal,red_dc_signal)
    ir_signal=np.divide(ir_ac_signal,ir_dc_signal)

    rr_signal=np.divide(red_signal, ir_signal)
    spO2=115-12.1*rr_signal
    for i in range(len(spO2)):
        if spO2[i]>100:
            spO2[i]=100
        
    return time_spO2, spO2, rr_signal

def find_heart_rate_from_CHROM(ts1, heart_rate_signal, intervall_length):

    peak_location=[]
    peaks,_ = find_peaks(heart_rate_signal, distance=6, height=0)
    for i in range(len(peaks)-1):
        peak_location.append(heart_rate_signal[peaks[i]])

    intervall_peak_numbers=[]  
    for i in range(int((len((heart_rate_signal))-intervall_length)/30)):
        result = np.where(np.logical_and(peaks>= 30*i, peaks<= intervall_length+30*i))[0]
        first_peak=peaks[result[0]]
        last_peak=peaks[result[-1]]
        intervall_peak_numbers.append((len(result)-1)*60/(last_peak-first_peak)*30)

    time_intervall_peak_numbers=np.linspace(ts1[intervall_length], ts1[-1],num=int((len((heart_rate_signal))-intervall_length)/30))
    return time_intervall_peak_numbers, intervall_peak_numbers


def find_valid_tidal_volumes(ts1, F_trend):
    breathing_rate=[]
    breath_timestamps=[]
    peak_location=[]

    peaks,_ = find_peaks(F_trend, width=9, distance=18, height=np.average(np.array(F_trend)))
    for i in range(len(peaks)-1):
        breathing_rate.append(1800/(peaks[i+1]-peaks[i]))
        breath_timestamps.append(ts1[peaks[i]])
        peak_location.append(F_trend[peaks[i]])
    


    tidal_volume=[]

    for i in range(len(peaks)-1):
        breath_start=peaks[i]
        breath_end=peaks[i+1]
        breath_signal=F_trend[breath_start:breath_end]
        maximum_depth=np.max(breath_signal)
        minimum_depth=np.min(breath_signal)
        tidal=maximum_depth-minimum_depth
        tidal_volume.append(tidal)
        

    tidal_volume=np.array(tidal_volume)


    #reject outliers in tidal volumes
    tidal_volume_sorted=np.sort(tidal_volume)
    volume_length=np.size(tidal_volume_sorted)
    mean=np.average(tidal_volume_sorted[int(volume_length/10):int(volume_length*9/10)])
    std=np.std(tidal_volume_sorted[int(volume_length/10):int(volume_length*9/10)])


    valid_tidal_volumes=[]
    valid_peaks=[]
    valid_timestamps=[]
    for i in range(len(tidal_volume)):
        if tidal_volume[i]>= 1 and tidal_volume[i]<=mean + 3*std and tidal_volume[i]>=mean -3*std:
            valid_tidal_volumes.append(tidal_volume[i])
            valid_timestamps.append(breath_timestamps[i])
            valid_peaks.append(peaks[i])

    return valid_timestamps, valid_peaks, valid_tidal_volumes, mean, std

def find_volume_and_rate_in_sliding_window(ts1, F_trend,valid_peaks, valid_tidal_volumes, intervall_length):
    intervall_volumes=[]
    volume_i=0
    valid_peaks=np.array(valid_peaks)  
    intervall_peak_numbers=[]  
    
    for i in range(int((len((F_trend))-intervall_length)/30)):
        result = np.where(np.logical_and(valid_peaks>= 30*i, valid_peaks<= intervall_length+30*i))[0]
        first_peak=valid_peaks[result[0]]
        last_peak=valid_peaks[result[-1]]
        if len(result)>1:
            intervall_peak_numbers.append((len(result)-1)*60/(last_peak-first_peak)*30)
            volume_i=0
            for j in range(len(result)-1):
                volume_i+=valid_tidal_volumes[result[j]]
            intervall_volumes.append(volume_i/(len(result)-1))
        else:
            if len(intervall_peak_numbers)>0:
                intervall_peak_numbers.append(intervall_peak_numbers[i-1])
                intervall_volumes.append(intervall_volumes[i-1])
            else:
                intervall_peak_numbers.append(np.nan)
                intervall_volumes.append(np.nan)



    time_intervall_peak_numbers=np.linspace(ts1[intervall_length], ts1[-1],num=int((len((F_trend))-intervall_length)/30))
    return time_intervall_peak_numbers, intervall_volumes, intervall_peak_numbers

def ventilator_volume_in_window(ventilator_volume_timestamps, ventilator_tidal_volume, intervall):
    ventilator_volume_timestamps=np.array(ventilator_volume_timestamps)
    vent_time_start=ventilator_volume_timestamps[0]
    vent_time_end=ventilator_volume_timestamps[-1]
    volume_windowed_values=[]
    volume_windowed_time=[]
    for i in range(int(vent_time_end-vent_time_start)):
        window_start=vent_time_start+i
        window_end=vent_time_start+i+int(intervall/30)
        result = np.where(np.logical_and(ventilator_volume_timestamps>= window_start, ventilator_volume_timestamps<= window_end))[0]
        volume_windowed_time.append(window_end)
        volume_windowed_values.append(np.mean(ventilator_tidal_volume[result[0]:result[-1]]))
    return volume_windowed_time, volume_windowed_values


def PCA_respiratory_signal(average_depth, ROI_x_1_array, ROI_x_2_array, ROI_y_1_array, ROI_y_2_array):
    butter= butter_bandpass(15/60, 120/60, 30, order=5)
    filtered_data=scipy.signal.sosfiltfilt(butter, average_depth)
    F_trend = PCA_timeseries(filtered_data)

    #Camera calibration parameters
    px=639.8630981445312
    py=370.0270690917969
    fx=611.5502319335938
    fy=611.1410522460938

    ROI_x1=np.multiply((np.array(ROI_x_1_array)-px), F_trend+np.mean(average_depth))/fx
    ROI_x2=np.multiply((np.array(ROI_x_2_array)-px), F_trend+np.mean(average_depth))/fx
    ROI_y1=np.multiply((np.array(ROI_y_1_array)-py), F_trend+np.mean(average_depth))/fy
    ROI_y2=np.multiply((np.array(ROI_y_2_array)-py), F_trend+np.mean(average_depth))/fy

    F_trend=F_trend*(ROI_x2-ROI_x1)*(ROI_y2-ROI_y1)/10**3
    return F_trend

def import_ventiliser_breath_starts(filename_ventiliser):                  
    breath_start=[]
    breath_end=[]
    with open(filename_ventiliser) as f: 
        for line in f:
            currentline = line.split(",") 
            if  currentline[0]!="breath_number":
                    start=currentline[1]
                    end=currentline[2]
                    breath_start.append(start)
                    breath_end.append(end)
    return breath_start, breath_end

def import_flow_from_ventilator(filename_ventilator_data):
    ventilator_flow=[]
    ventilator_time=[]
    with open(filename_ventilator_data) as f: 
            for line in f:
                currentline = line.split(",") 
                if  currentline[0]!='Time [ms]':
                            time=float(currentline[0])
                            time=time/1000
                            if currentline[2]!='':
                                        ventilator_flow.append(float(currentline[2]))
                                        ventilator_time.append(float(currentline[0]))
    return ventilator_time, ventilator_flow

def calculate_flow(ts1,F_trend):
    flow=[0]
    for i in range(len(F_trend)-1):
        time_0=ts1[i]
        time_1=ts1[i+1]
        volume_0=F_trend[i]
        volume_1=F_trend[i+1]
        flow.append((volume_1-volume_0)/(time_1-time_0)/1000)
    return flow


def extract_parameters_from_video(frame, depth, ir, params,file_poses_dict):
    img=cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    timestamp=params['timestamp']
    shifted_timestamps_poses = [x - timestamp for x in file_poses_dict["Timestamp"]]
    nearest_timestamp_index = np.abs(shifted_timestamps_poses).argmin()
    
    left_shoulder_x=file_poses_dict["Poses"][nearest_timestamp_index][0][0]
    left_shoulder_y=file_poses_dict["Poses"][nearest_timestamp_index][0][1]
    right_shoulder_x=file_poses_dict["Poses"][nearest_timestamp_index][1][0]
    right_shoulder_y=file_poses_dict["Poses"][nearest_timestamp_index][1][1]
    left_hip_x=file_poses_dict["Poses"][nearest_timestamp_index][2][0]
    left_hip_y=file_poses_dict["Poses"][nearest_timestamp_index][2][1]
    right_hip_x=file_poses_dict["Poses"][nearest_timestamp_index][3][0]
    right_hip_y=file_poses_dict["Poses"][nearest_timestamp_index][3][1]
    
    
    rectangle_y_1=int(min(left_shoulder_y, right_shoulder_y))
    rectangle_y_2=int(max(left_hip_y, right_hip_y))
    rectangle_x_2=int(max(left_hip_x, left_shoulder_x))
    rectangle_x_1=int(min(right_shoulder_x, right_hip_x))
    
        
    
    ROI_y_depth=int(rectangle_x_2)-int(rectangle_x_1)
    ROI_x_depth=int(rectangle_y_2)-int(rectangle_y_1)
    ROI_size_depth= ROI_x_depth *ROI_y_depth

    region_of_interest_data_red=np.ones(ROI_size_depth).reshape(ROI_y_depth,ROI_x_depth)
    region_of_interest_data_blue=np.ones(ROI_size_depth).reshape(ROI_y_depth,ROI_x_depth)
    region_of_interest_data_green=np.ones(ROI_size_depth).reshape(ROI_y_depth,ROI_x_depth)
    region_of_interest_data_depth=np.ones(ROI_size_depth).reshape(ROI_y_depth,ROI_x_depth)
    region_of_interest_data_ir=np.ones(ROI_size_depth).reshape(ROI_y_depth,ROI_x_depth)
    depth=cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ir=cv2.rotate(ir, cv2.ROTATE_90_COUNTERCLOCKWISE)

    masked_frame=skin_pixel_identification(img)
    for l in range(ROI_y_depth):
            for m in range(ROI_x_depth):
                region_of_interest_data_depth[l][m]=depth[l+int(rectangle_y_1)][m+int(rectangle_x_1)]     
                if masked_frame[l+int(rectangle_y_1)][m+int(rectangle_x_1)][2] !=0 and masked_frame[l+int(rectangle_y_1)][m+int(rectangle_x_1)][0] !=0 and masked_frame[l+int(rectangle_y_1)][m+int(rectangle_x_1)][1] !=0 and ir[l+int(rectangle_y_1)][m+int(rectangle_x_1)]!=0:
                    region_of_interest_data_red[l][m]=img[l+int(rectangle_y_1)][m+int(rectangle_x_1)][2]
                    region_of_interest_data_blue[l][m]=img[l+int(rectangle_y_1)][m+int(rectangle_x_1)][0]
                    region_of_interest_data_green[l][m]=img[l+int(rectangle_y_1)][m+int(rectangle_x_1)][1]
                    region_of_interest_data_ir[l][m]=ir[l+int(rectangle_y_1)][m+int(rectangle_x_1)]
                else:
                    region_of_interest_data_red[l][m]=0
                    region_of_interest_data_blue[l][m]=0
                    region_of_interest_data_green[l][m]=0
                    region_of_interest_data_ir[l][m]=0
    
    region_of_interest_data_depth[region_of_interest_data_depth==0]=np.nan
    region_of_interest_data_ir[region_of_interest_data_ir==0]=np.nan
    region_of_interest_data_red[region_of_interest_data_red==0]=np.nan
    region_of_interest_data_blue[region_of_interest_data_blue==0]=np.nan
    region_of_interest_data_green[region_of_interest_data_green==0]=np.nan
    
    average_depth=(np.nanmean(region_of_interest_data_depth))
    average_ir=(np.nanmean(region_of_interest_data_ir))/100
    average_red=np.nanmean(region_of_interest_data_red)
    average_blue=np.nanmean(region_of_interest_data_blue)
    average_green=np.nanmean(region_of_interest_data_green)
    
    return timestamp,average_ir, average_depth, average_red, average_blue, average_green, rectangle_x_1, rectangle_x_2, rectangle_y_1, rectangle_y_2 

def respiratory_rate_fourier(ts1, F_trend, intervall_length):
    fourier_breathing_rate=[]
    for i in range(int((len((F_trend))-intervall_length)/30)):
        window_signal=F_trend[30*i: 30*i + intervall_length]
        n = len(window_signal)
        t=np.arange(n)
        yf = rfft(window_signal)
        xf = rfftfreq(t.shape[-1], 1/1800)
        fourier_breathing_rate.append(xf[np.where(np.abs(yf)==np.max(np.abs(yf)))[0][0]])

    #plt.plot(xf, np.abs(yf))
    #plt.show()
    timestamps_fourier_resp_rate=np.linspace(ts1[intervall_length], ts1[-1],num=int((len((F_trend))-intervall_length)/30))
    return timestamps_fourier_resp_rate, fourier_breathing_rate



def Kalman_filter_1D(timeseries):

    kf = simdkalman.KalmanFilter(
        state_transition = np.array([[1,1],[0,1]]),
        process_noise = np.diag([0.01, 0.001]),
        observation_model = np.array([[1,0]]),
        observation_noise = 30.0)

    data=np.array(timeseries)

    # fit noise parameters to data with the EM algorithm (optional)
    kf = kf.em(data, n_iter=10)

    # smooth and explain existing data
    smoothed = kf.smooth(data)
    # predict new data
    pred = kf.predict(data, 50)
    # could be also written as
    # r = kf.compute(data, 15); smoothed = r.smoothed; pred = r.predicted

    smoothed_obs = smoothed.observations.mean
    return smoothed_obs

def POS_heart_rate_signal(red_average, green_average, blue_average):
    n=len(red_average) #signal length
    heart_rate_signal_POS=np.zeros(n)
    l=48 # window length
    for i in range(n):
        m=i-l+1
        if m>0:
            red=red_average[m:i]
            green=green_average[m:i]
            blue=blue_average[m:i]
            red= red- np.mean(red)
            green=green- np.mean(green)
            blue= blue- np.mean(blue)
            signal_1=green-blue
            signal_2=green+ blue - 2* red
            h=signal_1 + np.std(signal_1)/np.std(signal_2)* signal_2
            for j in range(len(h)):
                heart_rate_signal_POS[m+j]=heart_rate_signal_POS[m+j]+ h[j] - np.mean(h)
                
    butter= butter_bandpass(0.5, 4.5, 30, order=7)
    heart_rate_signal_POS=scipy.signal.sosfiltfilt(butter, heart_rate_signal_POS)
    return heart_rate_signal_POS


def Kalman_filter_2D(timeseries):

    kf = simdkalman.KalmanFilter(
    state_transition = np.array([[1,1],[0,1]]),
    process_noise = np.diag([0.2, 0.01])**2,
    observation_model = [[1,0],[1,0]],
    observation_noise = np.eye(2)*3**2)
    
    data=np.asarray(timeseries)

    # fit noise parameters to data with the EM algorithm (optional)
    #kf = kf.em(data, n_iter=5)

    # smooth and explain existing data
    smoothed = kf.smooth(data)
    # predict new data
    pred = kf.predict(data, 50)
    # could be also written as
    # r = kf.compute(data, 15); smoothed = r.smoothed; pred = r.predicted

    smoothed_obs = smoothed.observations.mean[0,:,0]
    return smoothed_obs


def rgb_oxygenation(ts1, filtered_red, filtered_blue, intervall):
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4574660/
    red_ac_signal,red_dc_signal=find_ac_and_dc_signal(filtered_red, intervall)
    blue_ac_signal,blue_dc_signal=find_ac_and_dc_signal(filtered_blue, intervall)
    time_spO2=np.linspace(ts1[intervall], ts1[-1],num=int((len((filtered_red))-intervall)/30))

    red_signal=np.divide(red_ac_signal,red_dc_signal)
    blue_signal=np.divide(blue_ac_signal,blue_dc_signal)

    rr_signal=np.divide(blue_signal, red_signal)
    spO2=spO2=40*np.log(rr_signal)+80
    for i in range(len(spO2)):
        if spO2[i]>100:
            spO2[i]=100
        
    return time_spO2, spO2, rr_signal

def ycgcr_oxygenation(ts1, red_average, green_average, blue_average, intervall):
    #https://www.mdpi.com/1424-8220/21/18/6120

    r_normalized=np.array(red_average)/255
    g_normalized=np.array(green_average)/255
    b_normalized=np.array(blue_average)/255

    Y=16+ (65.481 * r_normalized)+(128.533 * g_normalized)+(24.966 * b_normalized)
    Cg=128 + ( -81.085 * r_normalized)+(112 * g_normalized)+(-30.915 * b_normalized)
    Cr=128 + (112 * r_normalized)+(-93.786 * g_normalized)+(-18.214 * b_normalized)

    ac_filter= butter_bandpass(0.67, 4.5, 30, order=5)

    filtered_cg=scipy.signal.sosfiltfilt(ac_filter, Cg)+np.mean(Cg)
    filtered_cr=scipy.signal.sosfiltfilt(ac_filter, Cr)+np.mean(Cr)
    
    
    cg_ac_signal,cg_dc_signal=find_ac_and_dc_signal(filtered_cg, intervall)
    cr_ac_signal,cr_dc_signal=find_ac_and_dc_signal(filtered_cr, intervall)
    time_spO2=np.linspace(ts1[intervall], ts1[-1],num=int((len((filtered_cg))-intervall)/30))

    rr_ycgcr=np.divide(np.log(cr_ac_signal), np.log(cg_ac_signal))
    ycgcr_spO2=7.79* rr_ycgcr+88
    for i in range(len(ycgcr_spO2)):
        if ycgcr_spO2[i]>100:
            ycgcr_spO2[i]=100
        
    return time_spO2, ycgcr_spO2, rr_ycgcr