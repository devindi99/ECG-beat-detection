import wfdb
import numpy as np
import time
from beatdetection import beatpair

remove_sym = ["+", "|", "~", "x", "]", "[", "U", " MISSB", "PSE", "TS", "T", "P", "M", "\""]
slope_heights = []
sdiffs = []


def read_annotations(
        name: int,
        path: str) -> tuple:
    """

    :param name: name of the record as an integer
    :param path: folder path where the record exist
    :return: sample heights and the sampling frequency
    """

    path = path + str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    heights = [signals[i][0] for i in range(len(signals))]
    # resampled = signal.resample_poly(heights, 250, 360)
    fs = fields["fs"]
    return heights, fs


def max_min_slopes(
        n: int,
        samples: list,
        fs: int,
        threshold: int) -> tuple:
    """

    :param n: index of the current sample
    :param samples: sample heights
    :param fs: sampling frequency
    :return: maximum and minimum values of the slopes of the either side of the sample
    """
    global slope_heights
    a = round(0.027 * fs)

    slope_heights = []
    qrs_complex = False
    height_slope = 0
    slope_left = (samples[n] - samples[n - a]) / a
    slope_right = (samples[n] - samples[n + a]) / (-1 * a)
    s_mult = np.abs(slope_right * slope_left)
    state = (np.sign(slope_left) == -1 * np.sign(slope_right))
    # print("   "  , state)
    # print(threshold, "  ", s_mult)
    if s_mult > threshold and state:
        # print("cnudshdiv")
        height_slope = max(np.absolute(samples[n] - samples[n - a]), (np.absolute(samples[n + a] - samples[n])))
        slope_heights.append(height_slope)
        qrs_complex = True
    return s_mult, qrs_complex, height_slope


def third_criterion(
        cur_height: float,
        li: list) -> bool:
    """

    :param cur_height: slope height of the current sample
    :param li: list of slope heights
    :return: True -> sample validates third criterion, False -> otherwise
    """
    try:
        h_avg = np.average(np.absolute(li[-8:]))
    except IndexError:
        h_avg = np.average(np.absolute(li[:]))
    return np.abs(cur_height) > h_avg * 0.7


def locate_r_peaks(
        heights: list,
        fs: int) -> tuple:
    """
    :return: R peak locations and heights, time taken to complete the analysis
    """

    global slope_heights
    global sdiffs

    peaks = [0]
    locations = [0]
    slope_heights = []
    sdiffs = []

    count = 0
    start = time.time()
    # heights, fs = read_annotations(record, path)
    b = round(0.05 * fs)
    c = round(0.31875 * fs)
    a = round(0.03 * fs)
    s_mult = []

    for i in range(b, (b+3) * fs):
        smult, qrs_complex, height_slope = max_min_slopes(i, heights, fs,0)
        s_mult.append(smult)
    threshold = max(s_mult)/3

    for i in range((b+3)*fs, len(heights) + 1 - b):
        smult, qrs_complex_1, height_slope = max_min_slopes(i, heights, fs, threshold)
        state = third_criterion(height_slope, slope_heights)
        qrs_complex = qrs_complex_1 and state
        # print(state)
        if qrs_complex:

            if i - c > locations[-1]:
                locations.append(i)
                peaks.append(heights[i])
                s_mult.append(smult)
                slope_heights.append(height_slope)
            else:
                if smult > s_mult[-1]:
                    peaks[-1] = heights[i]
                    locations[-1] = i
                    slope_heights[-1] = height_slope
                    s_mult[-1] = smult
    # locations = locations[count:]
    # peaks = peaks[count:]

    end = time.time()

    return locations, peaks, end-start, count





