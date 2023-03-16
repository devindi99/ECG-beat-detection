import wfdb
import numpy as np
import time
from beatdetection import beatpair
remove_sym = ["+", "|", "~", "x", "]", "[", "U", " MISSB", "PSE", "TS", "T", "P", "M", "\""]
slope_heights = []
sdiffs = []


def initial(
        n: int,
        samples: list,
        fs: int) -> bool:
    """
    :param n: index of the current sample
    :param samples: sample heights
    :param fs: sampling frequency
    :return: True-> both first and second criterion is satisfied, False-> otherwise
    """
    global slope_heights
    global sdiffs

    second = False
    try:
        maximum_r, minimum_r, maximum_l, minimum_l, maximum_r_height, maximum_l_height = max_min_slopes(n, samples, fs)
        sdiff_max, max_height = max_slope_difference(maximum_r, minimum_r, maximum_l, minimum_l, maximum_l_height,
                                                     maximum_r_height)
        slope_heights.append(max_height)
        sdiffs.append(sdiff_max)
        if samples[n] > 20.480 / fs:
            teeta = 7.680 / fs
        elif 2.800 / fs < samples[n] < 20.480 / fs:
            teeta = 4.352 / fs
        else:
            teeta = 3.840 / fs

        first = sdiff_max > teeta

        if first:
            if maximum_l - minimum_r > maximum_r - minimum_l:
                smin = min(np.abs(maximum_l), np.abs(minimum_r))
                state = (np.sign(maximum_l) == -1 * np.sign(minimum_r))

            else:
                smin = min(np.abs(maximum_r), np.abs(minimum_l))
                state = (np.sign(maximum_r) == -1 * np.sign(minimum_l))

            if smin > 1.536 / fs and state:
                second = True
            else:
                second = False
        return first and second
    except ValueError:
        return False


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
        fs: int) -> tuple:
    """

    :param n: index of the current sample
    :param samples: sample heights
    :param fs: sampling frequency
    :return: maximum and minimum values of the slopes of the either side of the sample
    """
    a = round(0.027 * fs)
    b = round(0.063 * fs)
    lslopes = []
    rslopes = []
    lheights = []
    rheights = []
    for k in range(a, b + 1):
        if n + k >= len(samples) or n - k < 0:
            break
        lslopes.append((samples[n] - samples[n - k]) / k)
        rslopes.append((samples[n] - samples[n + k]) / (-1 * k))
        lheights.append(np.absolute(samples[n] - samples[n - k]))
        rheights.append(np.absolute(samples[n+k] - samples[n]))
    try:
        return max(rslopes), min(rslopes), max(lslopes), min(lslopes), max(rheights), max(lheights)
    except ValueError:
        return 0, 0, 0, 0, 0, 0


def max_slope_difference(
        maximum_r: float,
        minimum_r: float,
        maximum_l: float,
        minimum_l: float,
        maximum_l_height: float,
        maximum_r_height: float) -> tuple:
    """

    :param maximum_r: maximum slope to the right side of the sample
    :param minimum_r: minimum slope to the right side of the sample
    :param maximum_l: maximum slope to the left side of the sample
    :param minimum_l: minimum slope to the left side of the sample
    :param maximum_l_height: maximum slope height to the left of the sample
    :param maximum_r_height: maximum slope height to the right of the sample
    :return: maximum slope difference, maximum slope height
    """
    if maximum_l - minimum_r > maximum_r - minimum_l:
        return maximum_l - minimum_r, maximum_l_height
    else:
        return maximum_r - minimum_l, maximum_r_height


def teeta_diff(
        li: list,
        fs: int) -> float:
    """

    :param li: list of maximum slope differences
    :param fs: sampling frequency
    :return: threshold value to validate first criterion
    """
    s_avg = np.average(np.absolute(li[-8:]))

    if s_avg > 20.480 / fs:
        return 7.680 / fs
    elif 2.800 / fs < s_avg < 20.480 / fs:
        return 4.352 / fs
    else:
        return 3.840 / fs


def first_criterion(
        threshold: float,
        sdiff_max: float) -> bool:
    """

    :param threshold: threshold  value
    :param sdiff_max: current sample's maximum slope difference
    :return: True-> sample validates first criterion, False-> otherwise
    """
    return sdiff_max > threshold


def s_min(maximum_r: float,
          minimum_r: float,
          maximum_l: float,
          minimum_l: float) -> tuple:
    """

    :param maximum_r: maximum slope to the right side of the sample
    :param minimum_r: minimum slope to the right side of the sample
    :param maximum_l: maximum slope to the left side of the sample
    :param minimum_l: minimum slope to the left side of the sample
    :return: minimum of slopes, True-> sample is a local min/max False->otherwise
    """
    if maximum_l - minimum_r > maximum_r - minimum_l:
        smin = min(np.abs(maximum_l), np.abs(minimum_r))
        state = (np.sign(maximum_l) == -1 * np.sign(minimum_r))

    else:
        smin = min(np.abs(maximum_r), np.abs(minimum_l))
        state = (np.sign(maximum_r) == -1 * np.sign(minimum_l))
    return smin, state


def second_criterion(
        smin: float,
        state: bool,
        fs: int) -> bool:
    """

    :param smin: minimum of slopes
    :param state: True-> sample is a local min/max False->otherwise
    :param fs: sampling frequency
    :return: True -> sample validates second criterion, False -> otherwise
    """
    if smin > 1.536 / fs and state:
        return True
    else:
        return False


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
    return np.abs(cur_height) > h_avg * 0.4


def locate_r_peaks(
        heights: list,
        fs: int,
        n: int,
        ignore_afib: bool) -> tuple:
    """
    :param record: name of the record as an integer
    :param path: folder path where the record exist
    :param n: time (seconds) up to which the ECG data will be analysed
    :param ignore_afib: whether to ignore afib or not
    :return: R peak locations and heights, time taken to complete the analysis
    """

    global slope_heights
    global sdiffs

    peaks = []
    locations = []
    slope_heights = []
    sdiffs = []

    count = 0
    start = time.time()
    # heights, fs = read_annotations(record, path)
    b = round(0.063 * fs)
    c = round(0.31875 * fs)
    a = round(0.027 * fs)

    for i in range(b, 3 * fs):
        if initial(i, heights, fs):
            peaks.append(heights[i])
            locations.append(i)
            count += 1

    for i in range(b, len(heights) + 1 - b):
        try:
            # if ignore_afib:
            #     a_fib = beatpair.ref_annotate(record, path)[2]
            #     if i in a_fib:
            #         continue
            # print(i)
            maximum_r, minimum_r, maximum_l, minimum_l, maximum_r_height, maximum_l_height = max_min_slopes(i, heights,
                                                                                                        fs)
            sdiff_max, max_height = max_slope_difference(maximum_r, minimum_r, maximum_l, minimum_l, maximum_l_height,
                                                         maximum_r_height)
            teeta = teeta_diff(sdiffs, fs)
            smin, state = s_min(maximum_r, minimum_r, maximum_l, minimum_l)
            qrs_complex = first_criterion(teeta, sdiff_max) and second_criterion(smin, state, fs) and third_criterion(
                max_height, slope_heights)
            if qrs_complex:
                # element = max(np.absolute(heights[i - a:i + a + 1]))
                # loc = np.where(np.absolute(heights[i - a:i + a + 1]) == element)
                # loc = loc[0][0] + i - a
                if i - c > locations[-1]:
                    locations.append(i)
                    peaks.append(heights[i])
                    slope_heights.append(max_height)
                    sdiffs.append(sdiff_max)
                else:
                    if sdiff_max > sdiffs[-1]:
                        peaks[-1] = heights[i]
                        locations[-1] = i
                        slope_heights[-1] = max_height
                        sdiffs[-1] = sdiff_max
        except ValueError:
            continue
    locations = locations[count:]
    peaks = peaks[count:]
    end = time.time()

    return locations, peaks, end-start, count





