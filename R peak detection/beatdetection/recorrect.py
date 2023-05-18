from beatdetection import rpeakdetection
import numpy as np


def check_rr(sample1, sample2, d):
    if (sample2 - sample1 > d):
        return True
    else:
        return False


def check_peak(
        c: int,
        sub_list: list,
        fs: int,
        loc: int,
        end: int,
        peak: int,
        sd: list,
        sl: list,
        squared: list,
        denoised: list,
        sq: float,
        de: float) -> tuple:

    """

    :param c: the window that is considered as no two R peaks will be present
    :param sub_list: window that R peaks were not detected
    :param fs: sampling frequency
    :param loc: location of last R peak that was detected before the window where R peaks were absent
    :param end: location of the final R peak inside the window where R peaks were absent
    :param peak: height of the last peak before the window where R peaks were absent
    :param sd: sdiffs upto begin loc
    :param sl: slope heights upto begin loc
    :return: tuple
    """
    locations, peaks = rpeakdetection.new_r_peaks(sub_list, fs, c, loc, end, peak, sd, sl, squared, denoised, sq, de)
    # print(locations)
    return locations, peaks



