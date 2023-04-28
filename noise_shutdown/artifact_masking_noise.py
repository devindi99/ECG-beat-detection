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
from scipy.signal import *
import matplotlib.pyplot as plt
from noise_shutdown.SQI_noise import negentropy, compute_noise_ind

print_test = False


def rail_mask(raw_ecg):
    # Rail voltages are implemented to check for reading saturation which is mainly caused due to hardware level issues.
    # to be implemented with supply voltage of the input
    # not neeeded for scio cardio since it is checked at the hardware level.
    # different ECG acquisition hardware will have different rail voltages.
    return 0


def high_frequency_mask(raw_ecg, fs):
    """
    Identify masking segments with high frequency noise, above a threshold,
    to identify ecg segments severly corrupted by noise.
    :param raw_ecg: numpy array of ecg lead2 raw signal
    :param fs: sampling frequency of the input ecg signal
    :return: numpy array specifying noise for each ecg point
    """
    w0 = 50 / (fs / 2)
    w1 = 60 / (fs / 2)
    Q = 10
    b, a = iirnotch(w0, Q)
    b1, a1 = iirnotch(w1, Q)
    pl50_removed = filtfilt(b, a, raw_ecg)
    pl60_removed = filtfilt(b1, a1, pl50_removed)
    Wl = 40 / (fs / 2)
    b_e, a_e = ellip(5, 0.1, 40, Wl, btype="highpass")
    high_filtered = filtfilt(b_e, a_e, pl60_removed)
    x_2 = np.square(high_filtered)
    len_filter = int(0.05 * fs)
    h = firwin(len_filter, Wl)
    sn_1 = convolve(x_2, h, mode='same') / sum(
        h)  # the signal is low pass filtered using the normalized hamming window.
    selection_window = 1 * 60 * fs  # 1 minute data for window size,
    # (this window is taken into consideration based on the input segment size being 10 minutes).
    hw_sum = np.zeros(raw_ecg.shape)
    hw_std = np.zeros(raw_ecg.shape)
    hw_mask = np.zeros(raw_ecg.shape)
    for j in range(0, len(raw_ecg), selection_window): # The code base is similar to detecting peak
        start_ind = max(0, j)
        end_ind = min(j + selection_window, len(raw_ecg))
        x_in = sn_1[start_ind:end_ind]
        if len(x_in) == selection_window:
            threshold = 10
            peak_height = 0.05
            peaks = find_peaks(x_in, height=peak_height)[0]
            hw_sum[start_ind:end_ind] = np.sum(x_in)
            hw_std[start_ind:end_ind] = np.std(x_in)
            if len(peaks) > threshold:
                hw_mask[start_ind:end_ind] = 1
    return hw_mask, hw_sum, hw_std


def low_frequency_mask(raw_singal, fs):
    """
    Identify masking segments with low frequency noise, above a threshold
    :param raw_singal: numpy array of ecg lead2 raw signal
    :param fs: sampling frequency of the input ecg signal
    :return: numpy array specifying noise for each ecg point
    """

    wl = 0.7 / (fs / 2)
    wh = 10 / (fs / 2)
    b, a = iirfilter(N=5, Wn=[wl, wh], rp=0.1, rs=60, btype="bandpass", ftype="ellip")
    filtered_sig = filtfilt(b, a, raw_singal)
    x_2 = np.square(filtered_sig)
    len_filter = int(0.1 * fs)
    Wl = 20 / (fs / 2)
    h = firwin(len_filter, Wl)
    sn_1 = convolve(x_2, h, mode='same') / sum(h)
    selection_window = 1 * 60 * fs
    lw_mask = np.zeros(raw_singal.shape)
    threshold = 10
    peak_height = 8
    for j in range(0, len(raw_singal), selection_window): # The code base is similar to detecting peaks
        start_ind = max(0, j)
        end_ind = min(j + selection_window, len(raw_singal))
        x_in = sn_1[start_ind:end_ind]
        if len(x_in) == selection_window:
            peaks = find_peaks(x_in, height=peak_height)[0]
            if len(peaks) > threshold:
                lw_mask[start_ind:end_ind] = 1
    return lw_mask


def calculate_noise_mask_ecg(ecg_signal, fs):  # pragma: no cover
    """
    This function implements the
    :param ecg_signal:
    :param fs:
    :return:
    """
    # NOT USED IN CURRENT IMPLEMENTATION, REFER "find_noise_ind" function
    Vref = 2.4
    raw_ecg = ((ecg_signal / 2 ** 17) - 0.5) * (2 * Vref) / 3.5 * 1000
    hf_mask = high_frequency_mask(raw_ecg, fs)
    lf_mask = low_frequency_mask(raw_ecg, fs)
    final_mask = hf_mask + lf_mask

    noise_indices = np.asarray((final_mask >= 1) * 1)

    return noise_indices


def second_derv_noise(raw_signal, fs):
    """
    This estimation of noise from a segment provides information related to the muscle noise artifact.
    :param raw_signal: Raw ecg signal segment
    :param fs: Sampling frequency of the segment
    :return: Derivative mask for each sample raw_signal segment.
    """
    derv_mask = np.zeros(raw_signal.shape)
    d1 = np.diff(raw_signal)  # obtaining the first derivative of the input signal
    d2 = np.abs(np.diff(d1))  # obtaining the second derivative of the input signal
    wind_len = 10 * fs
    for j in range(2, len(raw_signal), wind_len):
        start_ind = j + 2
        end_ind = min(j + 1 + wind_len, len(raw_signal) - 1)
        derv_mask[start_ind:end_ind] = np.sum(d2[start_ind:end_ind])
    return derv_mask


def merge_noise_indicies(lw_mask, hf_mask, derv_mask, corr_value, neg_mask):
    """
    This function analyses each qrs complex to produce a final output to estimate the noise component in the complex
    :param lw_mask: Low frequency mask calculated using the "low_frequency_mask" function
    :param hf_mask: High frequency mask calculated using the "high_frequency_mask" function
    :param derv_mask: second derivative mask calculated from the second derivative estimation
    :param corr_value: The correlation based noise value assigned to the beat,
    please refer the file SQI.py for more information
    :return: The final noise estimation for the qrs peak.
    """
    if lw_mask == hf_mask:  # condition to check for easy final decision
        if lw_mask == 0:  # No significant low frequency or high frequency components
            if corr_value > 0.4:  # The R peak detection is significantly affected / possibility of different qrs
                # complexes between session
                if derv_mask >= 250:  # second derivative shows significant correlation with muscle noise
                    # this is checked to eliminate ectopies identified/misclassified as noise
                    return 1
                else:
                    if derv_mask >=150:
                        if neg_mask < 0.3:
                            return 1
                        else:
                            return 0
                    else:
                        if neg_mask <= 0.12 and derv_mask>=90:
                            return 1
                        else:
                            return 0
            else:  # if there is no significant change to the R peak detection
                # if derv_mask >= 180:  # very high second order derivative will imply,
                #     # presence of muscle noise in the qrs complex, this might reduce
                #     # the effectiveness of analyzing p or t wave morphology further
                #     return 1
                # else:
                if neg_mask <= 0.1:
                    if corr_value < 0.15 or derv_mask <= 80:
                        return 0
                    else:
                        return 1
                else:
                    if neg_mask < 0.25:
                        if corr_value > 0.3 and derv_mask >= 100:
                            return 1
                        else:
                            if corr_value > 0.35 and derv_mask >= 90:
                                return 1
                            else:
                                return 0
                    else:
                        return 0
        else:  # Significant low frequency and high frequency component in the segment
            if corr_value < 0.1:  # To check whether the significant frequency component
                # don't impact morphology or qrs location.
                return 0
            else:
                return 1
    else:
        if lw_mask == 0:  # low frequency components are insignificant, but high frequency components are present
            if corr_value >= 0.25:
                if derv_mask >= 130 and neg_mask < 0.2:
                    return 1
                else:
                    return 0
            else:
                if derv_mask >= 200 and neg_mask < 0.2:  # very high second order derivative will imply,
                    # presence of muscle noise in the qrs complex, this might reduce
                    # the effectiveness of analyzing p or t wave morphology further
                    return 1
                else:
                    if derv_mask >= 400:
                        return 1
                    else:
                        return 0
        else:  # high frequency components are insignificant, but low frequency components are present
            # if corr_value >= 0.4 or derv_mask >= 130:
            if (corr_value >= 0.35 or derv_mask >= 150) and neg_mask<0.2:
                return 1  # TO ensure that other random noise like variations are present in the signal segment
            else:
                if neg_mask < 0.1:
                    if corr_value < 0.15 or derv_mask <= 80:
                        return 0
                    else:
                        return 1
                else:
                    return 0


def merge_noise_indicies_negen(neg_mask, derv_mask, corr_value):
    """
    This function analyses each qrs complex to produce a final output to estimate the noise component in the complex
    :param lw_mask: Low frequency mask calculated using the "low_frequency_mask" function
    :param hf_mask: High frequency mask calculated using the "high_frequency_mask" function
    :param derv_mask: second derivative mask calculated from the second derivative estimation
    :param corr_value: The correlation based noise value assigned to the beat,
    please refer the file SQI.py for more information
    :return: The final noise estimation for the qrs peak.
    """
    if corr_value < 0.1:
        return 0
    else:
        if corr_value >= 0.4:
            if derv_mask >= 200:
                return 1
            else:
                if neg_mask <= 0.15:
                    return 1
                else:
                    return 0
        else:
            if neg_mask <= 0.1:
                return 1
            else:
                if derv_mask >= 250:
                    return 1
                else:
                    return 0 # high frequency components are insignificant, but low frequency components are present


def negen_mask(ecg, fs):
    window_size = 10
    return_mask = np.zeros(ecg.shape)
    windows_len = int(window_size*fs)
    for j in range(0,len(ecg),windows_len):
        start_ind = max(0,j)
        end_ind = min(len(ecg),j+windows_len)
        ecg_seg = ecg[start_ind:end_ind]
        return_mask[start_ind:end_ind] = negentropy(ecg_seg)

    return return_mask


def final_noise_ind(raw_ecg, fs, peak_indicies, corr_val):
    """
    This function provides the final noise annotation for each qrs peak within the ecg segment.
    :param raw_ecg: Input ECG segment in mV
    :param fs: Sampling frequency of the signal
    :param peak_indicies: The peaks identified in the signal
    :param corr_val: The correlative noise value assigned for each peak from SQI implementation.
    :return:
    """
    global print_test
    if (raw_ecg.ndim == 1) and (raw_ecg.shape == (len(raw_ecg),)):
        hf_mask, hf_sum, hf_std = high_frequency_mask(raw_ecg, fs)
        lf_mask = low_frequency_mask(raw_ecg, fs)
        derv_mask = second_derv_noise(raw_ecg, fs)
        final_mask = np.zeros(np.shape(peak_indicies))
        neg_mask = negen_mask(raw_ecg,fs)
        if len(peak_indicies) > 0:
            hf_peaks = hf_mask[peak_indicies]  # converting the segment mask and annotating to each peak index
            lf_peaks = lf_mask[peak_indicies]  # converting the segment mask and annotating to each peak index
            derv_peaks = derv_mask[peak_indicies]  # converting the segment mask and annotating to each peak index
            neg_peaks = neg_mask[peak_indicies]
            merging_windows = 10
            mergin_window_size = merging_windows * fs
            for j in range(0, len(raw_ecg), mergin_window_size): # for each qrs complex the merging is done individually
                start_index = j
                end_index = min(j + mergin_window_size, len(raw_ecg) - 1)
                seg_lf = lf_peaks[(peak_indicies >= start_index) & (peak_indicies < end_index)]
                seg_hf = hf_peaks[(peak_indicies >= start_index) & (peak_indicies < end_index)]
                seg_derv_mask = derv_peaks[(peak_indicies >= start_index) & (peak_indicies < end_index)]
                seg_corr = corr_val[(peak_indicies >= start_index) & (peak_indicies < end_index)]
                seg_negen_mask = neg_peaks[(peak_indicies >= start_index) & (peak_indicies < end_index)]
                seg_final_mask = np.zeros(np.shape(seg_lf))
                if len(seg_final_mask) > 0:
                    for i in range(0, len(seg_lf)):
                        seg_final_mask[i] = merge_noise_indicies(seg_lf[i], seg_hf[i], seg_derv_mask[i], seg_corr[i], seg_negen_mask[i])
                    if sum(seg_final_mask) / len(seg_final_mask) >= 0.5:
                        final_mask[(peak_indicies >= start_index) & (peak_indicies < end_index)] = np.ones(np.shape(seg_lf))
                    else:
                        final_mask[(peak_indicies >= start_index) & (peak_indicies < end_index)] = np.zeros(
                            np.shape(seg_lf))
            if print_test: # pragma: no cover
                fig1 = plt.figure(1)
                plt.subplot(7, 1, 1)
                plt.plot(raw_ecg)
                plt.subplot(7, 1, 2)
                plt.plot(hf_mask)
                plt.subplot(7, 1, 3)
                plt.plot(lf_mask)
                plt.subplot(7, 1, 4)
                plt.plot(derv_mask)
                plt.subplot(7, 1, 5)
                plt.plot(peak_indicies, corr_val)
                plt.subplot(7, 1, 6)
                plt.plot(peak_indicies, final_mask)
                plt.subplot(7, 1, 7)
                plt.plot(neg_mask)
                plt.title(f'Shutdown percentage {np.sum(final_mask)*100/np.size(final_mask):0.3f}%')
                fig2 = plt.figure(2)
                plt.subplot(4, 1, 1)
                plt.plot(raw_ecg)
                plt.subplot(4, 1, 2)
                plt.plot(hf_mask)
                plt.subplot(4, 1, 3)
                plt.plot(hf_sum)
                plt.subplot(4, 1, 4)
                plt.plot(hf_std)
                plt.show()
        return final_mask
    else:
        raise ValueError('Incorrect input ECG segment shape, ensure input in the shape (m,)')


def identify_noise(raw_ecg, filtered_ecg, peaks, fs=250):
    """
    The function detects noise in the ECG segment using correlation, entropy, low_frequency_mask, high_frequency_mask, derv_mask.
    :param raw_ecg: numpy array containing raw ECG segment in mV, sampled at 250Hz, and greater than 40 s
    :param filtered_ecg: numpy array containing filtered ECG segment in mV, sampled at 250Hz, and greater than 40 s
    :param peaks: Peak locations of the ECG in samples
    :param fs: frequency(Hz) at which the ECG is sampled at, default = 250Hz
    :return: returns an array indicating whether each peak is noisy or not
    """
    minimum_duration = 20 # duration of ECG required in sec
    if fs != 250:
        import warnings
        warnings.warn("The input ECG signal must be sampled at 250Hz for the code to function as intended")
    if len(raw_ecg) != len(filtered_ecg):
        raise ValueError('The duration of raw and filtered ECG signal segments should be the same.')
    else:
        if len(raw_ecg) < minimum_duration * fs:
            raise ValueError(f"The duration of the input ECG segment must be greater than {minimum_duration}s.")
        else:
            noise_segment_len = 5 # in sec
            segment_noise_ind, _ = compute_noise_ind(filtered_ecg, peaks, noise_segment_len, fs)
            return final_noise_ind(raw_ecg,fs,peaks,segment_noise_ind)



