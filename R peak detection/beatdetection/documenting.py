from typing import Tuple, List

from beatdetection import rpeakdetection
import numpy as np
from beatdetection import beatpair
from beatdetection import recorrect
from openpyxl import Workbook
import matplotlib.pyplot as plt
import time
from scipy import signal
import wfdb
from beatdetection import filters

# Filter requirements.
T = 1/250         # Sample Period
fs = 250       # sample rate, Hz
cutoff = 30      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2     # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def read_annotations(
        name: int,
        path: str) -> Tuple[List[float], int]:
    """
    ECG signal is extracted from the data base using WFDB tool box and passed through a low pass filter and two cascaded
    median filters to remove high frequency noise and baseline wander.

    :param name: name of the record as an integer
    :param path: folder path where the record exist
    :return: low pass filtered ECG signal and the sampling frequency (250 Hz)
    """

    path = path + str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    heights = [signals[i][0] for i in range(len(signals))]
    resampled = signal.resample_poly(heights, 250, 360)
    fs = 250
    t = [i for i in range(len(resampled))]
    # plt.subplot(2, 1, 1)
    plt.plot(t, resampled)
    # plt.title("Original ECG signal")

    y = filters.butter_lowpass_filter(resampled, cutoff, fs, order)
    t = [i for i in range(len(y))]
    plt.plot(t,y)
    # Resample the original ECG signal to 250 Hz, comment out when using AHA database
    heights = filters.Low_pass(resampled)

    t = [i for i in range (len(heights))]

    # plt.subplot(2, 1, 2)
    plt.plot(t, heights)
    # plt.title("Low pass filtered signal")
    # baseline removal using two cascaded median filters
    QRS_removed = signal.medfilt(heights, kernel_size=round(0.2 * 250) + 1)  # Remove QRS and P waves
    T_removed = signal.medfilt(QRS_removed, kernel_size=round(0.6 * 250) + 1)  # Remove T waves
    heights = heights - T_removed

    # A high pass filter for removing baseline wander, can use this instead of the two median filters
    # heights = filters.iir(heights, 2000)

    return heights, fs


for record in range(105, 235):

    try:
        remove = [102, 104, 107, 217]
        if record in remove:
            continue
        path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
        heights, fs = read_annotations(record, path)
        plt.show()
        # t = [i for i in range(len(heights))]
        # plt.plot(t, heights)
        print(record)


    except FileNotFoundError:
        continue
