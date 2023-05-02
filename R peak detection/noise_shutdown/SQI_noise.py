#  Copyright © Synergen Technology Labs LLC (https://synergentl.com/)
#  All Rights Reserved.
#  NOTICE: All information contained herein is, and remains
#  the property of Synergen Technology Labs LLC and its suppliers,
#  if any. The intellectual and technical concepts contained
#  herein are proprietary to Synergen Technology Labs LLC
#  and its suppliers and may be covered by United States and Foreign Patents or
#  patents in process and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material
#  is strictly forbidden unless prior written permission is obtained
#  from Synergen Technology Labs LLC

import numpy as np
from scipy.stats import pearsonr
import math

QRS_avg_n = []
QRS_corr = 0


def compute_rr(peaks):
    """
    This function creates an array containing all the RR interval length provided in samples
    :param peaks: an array consisting the x location of the peaks
    :return: rr - A numpy array consisting the R-R interval between consecutive peaks.
     If len(peaks) is N, then len(RR) is N-1
    """
    rr = []
    for i in range(0, len(peaks) - 1):
        rr.append(peaks[i + 1] - peaks[i])
    return np.asarray(rr)


def rr_beats_check(x_arr, fs, peaks):
    """
    This function checks the RR beats interval,
    this function compares whether all RR interval falls below a certain threshold (in time).
    :param x_arr: The input ECG signal array
    :param fs: sampling frequency of the signal
    :param peaks: identified R-peaks location
    :return: True; if the RR_interval has satisfied the condition, False; otherwise
    RR: array consisting of all the RR interval values
    """
    rr = compute_rr(peaks)
    rr_t = np.asarray([i / fs for i in rr])
    max_hr_variation_in_seg = 6
    rr_max = max(rr)
    rr_min = min(rr)
    ratio = rr_max / rr_min
    if ratio > max_hr_variation_in_seg:
        return False, rr
    else:
        return True, rr


def avg_correlation(qrs_avg, qrs_val):
    """
    This function calculates the average correlation between the Average QRS complex,
    and list of all the QRS complex in the signal.
    :param qrs_avg: Numpy array of the average QRS complex.
    :param qrs_val: List of all QRS complexes in the ECG signal array
    :return: Average pearson correlation between all QRS complex and the average QRS complex
    """
    r = []
    for i in range(0, len(qrs_val)):
        corr, prob = pearsonr(qrs_avg, qrs_val[i])
        corr = abs(corr)
        r.append(corr)
    return sum(r) / len(r), r


def template_match(x_arr, rr, fs, peaks):
    """
    This function computes the average QRS complex, and evaluates the similarity of the average QRS and all other QRS complex.
    The similarity measure is computed using the pearson correlation coefficient.
    :param x_arr: numpy array of the ECG signal
    :param rr: List containing the values of all successive RR interval
    :param fs: The sampling frequency of the signal in Hz. (float)
    :param peaks: numpy array consisting the location of all R peaks in the ECG signal
    :return: True: if the similarity measure between the average QRS complex and all other QRS complex is high,
    False; otherwise.
    """
    global QRS_corr
    global QRS_avg_n
    med_rr = np.median(rr)  # median RR interval is preferred rather than mean because mean RR interval,
    # can be affected by outliers
    qrs_comp = []
    qrs_avg = 0
    for i in range(0, len(peaks)):
        low_xval = max(peaks[i] - int(med_rr / 2), 0)
        high_xval = min(peaks[i] + int(med_rr / 2), len(x_arr) - 1)
        x_arr_seg = np.asarray(x_arr[low_xval:high_xval])
        # following conditions are checked to create QRS_segments of same length padded with zeros
        if (peaks[i] - int(med_rr / 2)) < 0:
            padd_amount = -1 * (peaks[i] - int(med_rr / 2))
            x_arr_seg = np.concatenate((np.ravel(np.zeros((padd_amount, 1))), x_arr_seg), axis=0)
        elif (peaks[i] + int(med_rr / 2)) >= len(x_arr):
            padd_amount = peaks[i] + int(med_rr / 2) - len(x_arr) + 1
            x_arr_seg = np.concatenate((x_arr_seg, np.ravel(np.zeros((padd_amount, 1)))), axis=0)
        if i == 0:
            qrs_avg = x_arr_seg / len(peaks)
        else:
            qrs_avg += x_arr_seg / len(peaks)
        qrs_comp.append(x_arr_seg)

    coeff_avg, qi_array = avg_correlation(qrs_avg, qrs_comp)  # computing the average correlation,
    QRS_corr = coeff_avg
    QRS_avg_n = qrs_avg
    return coeff_avg, qi_array


def compute_noise_satistically(peak_loc):
    """
    This implementation is only concerning segments of number of peaks less than 3.
    Here some statistical values are extracted and compared to compute the noise quantification of the segment without QRS complexes.
    """
    if len(peak_loc) == 0:
        return 0, [], []
    elif len(peak_loc) == 1:
        return 0, [], [2]


def __differential_entropy(seg_hist):
    """
    This method computes the differential entropy of the signal based on the equation,
    {\displaystyle S(p_{x})=-\int p_{x}(u)\log p_{x}(u)\,du}
    :param X: X is the input signal segment
    :return: The differential entropy value
    """
    sum_val = 0

    for i in range(0, len(seg_hist)):
        try:
            sum_val += math.log(seg_hist[i],2) * seg_hist[i]
        except ValueError:  # values to close to zero can throw errors thus adding zeros automatically for them
            sum_val += 0
    return -1 * sum_val


def negentropy(data):
    """
    This method calculates the negative entropy of the signal.
    i.e. how much the signal is similar to a gaussian signal of same segment length and std.
    :param data: the signal segment to be compared and calculated for negative entropy.
    :return: negative entropy value
    """
    std = np.std(data)
    mean_val = np.mean(data)
    gx = np.random.normal(mean_val,std,data.shape)
    hist_gx, edge_gx = np.histogram(gx,
                                    bins=50)  # obtaining the histogram for computing the probability dispersion of
    # values
    hist_x, edge_x = np.histogram(data, bins=50)
    hist_gx = hist_gx / np.sum(hist_gx)  # probability derived by dividing by total number of n
    hist_x = hist_x / np.sum(hist_x)

    return __differential_entropy(hist_gx) - __differential_entropy(hist_x)


def get_qrs_quality(x_arr, fs, peaks):
    """
    This function is called, to identify whether the signal array x_arr is noisy or non-noisy.
    :param x_arr: Array consisting the input ECG signal
    :param fs: The sampling frequency of the signal
    :param peaks: Location of all the R-peaks in the signal.
    :return:
    """
    global QRS_avg_n
    use_heart_rate_for_feasibility_check = False  # Specifies the use of Heart rate for feasibility check
    if len(peaks) < 2:
        return compute_noise_satistically(peaks)
    else:
        if use_heart_rate_for_feasibility_check:
            hr = (len(peaks) - 1) * fs * 60 / (peaks[-1] - peaks[0])
            low_hr_thresh = 40  # minimum possible heart rate
            high_hr_thresh = 300  # maximum possible heart rate

            if low_hr_thresh <= hr <= high_hr_thresh:  # THIS line checks whether the Heart rate(BPM),
                #  computed from the peaks lie in a reasonable range
                bool_chk, rr = rr_beats_check(x_arr, fs, peaks)
                if bool_chk:
                    corr, corr_arr = template_match(x_arr, rr, fs, peaks)
                    return corr, QRS_avg_n, corr_arr
                else:
                    corr_arr = np.zeros(peaks.size)
                    return 0, QRS_avg_n, corr_arr
            else:
                corr_arr = np.zeros(peaks.size)
                return 0, QRS_avg_n, corr_arr
        else:
            bool_chk, rr = rr_beats_check(x_arr, fs, peaks)
            if bool_chk:
                corr, corr_arr = template_match(x_arr, rr, fs, peaks)
                return corr, QRS_avg_n, corr_arr
            else:
                corr_arr = np.zeros(peaks.size)
                return 0, QRS_avg_n, corr_arr


def compute_noise_ind(filtered_signal, peak_location, size_in_time, fs):
    """
    Implemented for integration with the rpeakfilter module
    :param filtered_signal:
    :param peak_location: The location of peaks specified in samples
    :param size_in_time: The input signal will be broken down into segments of the size specified to be analysed
    :param fs: The sampling rate of the input signal
    :return: The noise index array.
    """
    # the sampling rate of the device is 250Hz
    peak_samples = np.asarray(peak_location)
    segment_samples = size_in_time * fs
    x_arr = np.asarray(filtered_signal)
    segment_noise_arr = np.zeros(peak_samples.size)
    segment_noise_vals = np.zeros(peak_samples.size)
    if len(peak_location) > 0:
        for start_ind in range(0, x_arr.size, segment_samples):
            end_ind = min(x_arr.size, start_ind + segment_samples)
            x_in = x_arr[start_ind:end_ind]
            peaks_i = peak_samples[(peak_samples >= start_ind) & (peak_samples < end_ind)]
            if x_in.size == segment_samples:
                segment_sqi, avg_qrs_complex, peaks_sqi = get_qrs_quality(x_arr, fs, peaks_i)
                segment_noise = 1 - segment_sqi
                noise_val_segment = np.asarray([segment_noise for i in range(0, len(peaks_sqi))])
                segment_noise_vals[(peak_samples >= start_ind) & (peak_samples < end_ind)] = peaks_sqi
                segment_noise_arr[(peak_samples >= start_ind) & (peak_samples < end_ind)] = noise_val_segment
    return segment_noise_arr, segment_noise_vals


if __name__ == '__main__': # pragma: no cover
    """
    code snippet to test whether the module is functioning as intended.
    comment this segment if the implementation is not under testing
    """
    import random

    segment_size = 10
    fs = 250
    X_size = segment_size * fs
    input_x = np.random.rand(X_size)
    peak_rand = [random.randrange(X_size), random.randrange(X_size)]
    peaks = [min(peak_rand), max(peak_rand)]
    noise_ind = compute_noise_ind(input_x, peaks, segment_size, fs)
    print(noise_ind)
