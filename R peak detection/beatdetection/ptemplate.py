import numpy as np
import pywt
from beatdetection import rpeakdetection
from beatdetection import filters

# records = [16265, 16272, 16273, 16420, 16773, 16539, 16786, 17152, 17453]

db1 = pywt.Wavelet('haar')
template = []
p_wave = []


def qrs(
        r: np.ndarray,
        n: int) -> int:

    """

    :param r: R peak locations
    :param n: R peak location of considered beat
    :return: onset location
    """
    w = 30
    n = int(n)
    wn1 = r[int(max(n - w, 0)):n]
    wn2 = r[n:int(min(n + w, len(r)))]
    """searching for the first half"""
    l_min = min(wn1)
    lmax = max(wn2)
    la = abs(lmax - l_min)
    l_min += (la / 100)
    lmax -= (la / 100)
    qb = n - (len(wn1) - max(np.where(wn1 <= l_min)[-1])) - 5
    return qb


def extract_p_waves(
        record: int,
        path: str,
        n: int) -> None:

    """

    :param record: name of the record as an integer
    :param path: folder path where the record exist
    :param n: time (seconds) upto which the ECG data will be analysed
    :return: None
    """

    global template
    global p_wave

    try:
        locations, peaks = rpeakdetection.locate_r_peaks(record, path, n)
        heights, fs = rpeakdetection.read_annotations(record, path)
        size = 100
        y1 = filters.Low_pass(heights)
        y = filters.iir(y1, 100)
        Base = np.mean(y)
        ybp = filters.high_pass(y)
        ysq = filters.sqr_fil(filters.l_deriv(ybp - Base))
        ymv = filters.mov_win(ysq)
        ydr = filters.l_deriv(ymv)

        for i in range(len(locations) - 1):
            onset_point = qrs(ydr, locations[i + 1])
            rr = locations[i + 1] - locations[i]
            second_half = heights[(locations[i] + rr // 2):onset_point + 1]
            out = np.linspace(locations[i] + rr // 2, onset_point + 1, size)
            xp = [m for m in range((locations[i] + rr // 2), onset_point + 1)]
            yp = second_half
            second_half = np.interp(out, xp, yp)
            ca2, cd2, cd1 = pywt.wavedec(second_half, db1, mode='constant', level=2)
            template.append(ca2)

            maximum = np.max(ca2)
            minimum = np.min(ca2)
            z = np.array([(x - minimum) / (maximum - minimum) for x in ca2])
            p_wave.append(z)
    except FileNotFoundError:
        print("File: ", record, "doesn't exist.")
    return None


def create_template(
        records: list,
        path: str,
        time: int) -> np.ndarray:

    """

    :param records: records that will be used to create the template
    :param path: folder path where the records exist
    :param time: time (seconds) upto which the ECG data will be analysed
    :return: template wave
    """
    global template
    template = []

    for record in records:
        extract_p_waves(record, path, time + 4)

    avg = np.average(template, axis=0)
    maximum = np.max(avg)
    minimum = np.min(avg)
    z = np.array([(x - minimum) / (maximum - minimum) for x in avg])
    return z


def p_waves(
        record: int,
        path: str,
        time: int) -> list:

    """

    :param record: name of the record as an integer
    :param path: folder path where the record exist
    :param time: time (seconds) upto which the ECG data will be analysed
    :return: p waves of the record as a 2D array
    """
    global p_wave
    p_wave = []
    extract_p_waves(record, path, time)

    return p_wave
