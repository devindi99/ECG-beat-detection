import wfdb
import numpy as np
from BaselineRemoval import BaselineRemoval
from beatdetection import filters
from scipy import signal

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

        if samples[n] > 20.480 / fs:
            teeta = 9.680 / fs
        elif 2.800 / fs < samples[n] < 20.480 / fs:
            teeta = 6.352 / fs
        else:
            teeta = 5.840 / fs

        first = sdiff_max > teeta

        if first:
            if maximum_l - minimum_r > maximum_r - minimum_l:
                smin = min(np.abs(maximum_l), np.abs(minimum_r))
                state = (np.sign(maximum_l) == -1 * np.sign(minimum_r))

            else:
                smin = min(np.abs(maximum_r), np.abs(minimum_l))
                state = (np.sign(maximum_r) == -1 * np.sign(minimum_l))

            if smin > 2.536 / fs and state:
                second = True
            else:
                second = False
        if first and second:
            slope_heights.append(max_height)
            sdiffs.append(sdiff_max)
        return first and second
    except ValueError:
        return False


def read_annotations(
        name: int,
        path: str) -> tuple:
    """

    :param name: name of the record as an integer
    :param path: folder path where the record exist
    :return: low pass filtered ECG signal and the sampling frequency (250 Hz)
    """

    path = path + str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    heights = [signals[i][0] for i in range(len(signals))]
    resampled = signal.resample_poly(heights, 250, 360)

    # Resample the original ECG signal to 250 Hz, comment out when using AHA database
    heights = filters.Low_pass(resampled)

    # A high pass filter for removing baseline wander, can use this instead of the two median filters
    # heights = filters.iir(heights, 2000)
    fs = 250
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
    :return: a tuple contatining maximum slope difference and maximum slope height
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
        return 9.680 / fs
    elif 2.800 / fs < s_avg < 20.480 / fs:
        return 6.352 / fs
    else:
        return 5.840 / fs


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
    if smin > 2.536 / fs and state:
        return True
    else:
        return False


def second_criterion_re(
        smin: float,
        state: bool,
        fs: int) -> bool:
    """

    :param smin: minimum of slopes
    :param state: True-> sample is a local min/max False->otherwise
    :param fs: sampling frequency
    :return: True -> sample validates second criterion, False -> otherwise
    """
    if smin > 10.36 / fs and state:
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


def third_criterion_re(
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
    return np.abs(cur_height) > h_avg * 0.3

def locate_r_peaks(
        heights: list,
        fs: int,
        c: int,
        callibrate: bool,
        lo: list,
        pe: list,
        sl: list,
        sd: list) -> tuple:

    """

    :param heights: sample heights
    :param fs: sampling frequency
    :param c: window that is considered as the detected R peaks belong to the same R peak
    :param callibrate: True -> a callibration was done beforehand, False -> otherwise
    :param lo: The R peak locations detected so far
    :param pe: The R peak heights detected so far
    :param sl: slope height values detected so far
    :param sd: slope difference values detected so far
    :return:
    """

    global slope_heights
    global sdiffs

    peaks = pe.copy()
    locations = lo.copy()
    slope_heights = sl.copy()
    sdiffs = sd.copy()

    b = round(0.063 * fs)
    a = round(0.027 * fs)

    if callibrate:
        f = round(5*60*fs)+1
    else:
        f = b

    if len(locations) == 0:
        for i in range(f, f+(3 * fs)):
            if initial(i, heights, fs):
                element = max(np.absolute(heights[i - b:i + b + 1]))
                loc = np.where(np.absolute(heights[i - b:i + b + 1]) == element)
                loc = loc[0][0] + i - b

                if loc not in locations:
                    peaks.append(heights[loc])
                    locations.append(loc)

    count = len(locations)
    m = f

    while m < len(heights) + 1 - b:
        try:
            maximum_r, minimum_r, maximum_l, minimum_l, maximum_r_height, maximum_l_height = max_min_slopes(m, heights,
                                                                                                        fs)
            sdiff_max, max_height = max_slope_difference(maximum_r, minimum_r, maximum_l, minimum_l, maximum_l_height,
                                                         maximum_r_height)
            teeta = teeta_diff(sdiffs, fs)
            smin, state = s_min(maximum_r, minimum_r, maximum_l, minimum_l)
            qrs_complex = first_criterion(teeta, sdiff_max) and second_criterion(smin, state, fs) and third_criterion_re(
                max_height, slope_heights)

            if qrs_complex:
                element = max(np.absolute(heights[m - a:m + a + 1]))
                loc = np.where(np.absolute(heights[m - a:m + a + 1]) == element)
                loc = loc[0][0] + m - a
                if loc - c > locations[-1]:
                    locations.append(loc)
                    peaks.append(heights[loc])
                    slope_heights.append(max_height)
                    sdiffs.append(sdiff_max)
                    m = loc


                else:
                    if sdiff_max > sdiffs[-1]:
                        peaks[-1] = heights[loc]
                        locations[-1] = loc
                        slope_heights[-1] = max_height
                        sdiffs[-1] = sdiff_max
                        m = loc

            m += 1

        except ValueError:
            continue

    if not callibrate:
        count = count-1

    return locations[count:], peaks[count:], count, sdiffs[count:], slope_heights[count:]


def new_r_peaks(
        heights: list,
        fs: int,
        c: int,
        begin_loc: int,
        end_loc: int,
        begin_peak: int,
        sd: list,
        sl: list) -> tuple:

    """

    :param heights: list of heights
    :param fs: sampling frequency
    :param c: the window that is considered as no two R peaks will be present
    :param begin_loc: location of last R peak that was detected before the interval where no R peak was detected
    :param end_loc: location of the final R peak after the interval where no R peak was detected
    :param begin_peak: height of the last peak before the interval where no R peak was detected
    :param sd: slope difference values up to begin_loc
    :param sl: slope heights upt to begin_loc
    :return: newly detected R peaks
    """

    global slope_heights
    global sdiffs

    peaks = [begin_peak]
    locations = [begin_loc]
    slope_heights = sl
    sdiffs = sd
    b = round(0.063 * fs)
    a = round(0.027 * fs)

    for i in range(b, len(heights) + 1 - b):

        try:

            maximum_r, minimum_r, maximum_l, minimum_l, maximum_r_height, maximum_l_height = max_min_slopes(i, heights,
                                                                                                            fs)
            sdiff_max, max_height = max_slope_difference(maximum_r, minimum_r, maximum_l, minimum_l, maximum_l_height,
                                                         maximum_r_height)
            teeta = teeta_diff(sdiffs, fs)
            smin, state = s_min(maximum_r, minimum_r, maximum_l, minimum_l)
            qrs_complex = first_criterion(teeta, sdiff_max) and second_criterion_re(smin, state, fs) and third_criterion_re(
                max_height, slope_heights)

            if qrs_complex:
                element = max(np.absolute(heights[i - a:i + a + 1]))
                loc = np.where(np.absolute(heights[i - a:i + a + 1]) == element)
                loc = loc[0][0] + i - a
                if loc+begin_loc + c < end_loc:
                    if loc+begin_loc - c > locations[-1]:
                        locations.append(loc+begin_loc)
                        peaks.append(heights[loc])
                        slope_heights.append(max_height)
                        sdiffs.append(sdiff_max)

                    else:
                        if sdiff_max > sdiffs[-1]:
                            peaks[-1] = heights[loc]
                            locations[-1] = loc+begin_loc
                            slope_heights[-1] = max_height
                            sdiffs[-1] = sdiff_max

        except ValueError:
            continue
    locations = locations[1:]
    peaks = peaks[1:]

    return locations, peaks


