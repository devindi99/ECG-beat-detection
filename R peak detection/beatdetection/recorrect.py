"""
Module for checking signal sections with no R peak detected
"""
from typing import Tuple

from beatdetection import rpeakdetection


def check_rr(
        sample1: int,
        sample2: int,
        rr_int: float) -> bool:

    """
    Check if a certain signal section has a possibility of containing an undetected R peak.

    :param sample1: Location of previous R peak
    :param sample2: Location of R peak
    :param rr_int: Estimated RR interval
    :return: True -> The detected RR interval is larger than the estimated. There is a possibility of a R peak existing
    in between the two R peaks, False -> otherwise
    """
    if sample2 - sample1 > rr_int:
        return True
    else:
        return False


def check_peak(
        c: int,
        sub_list: list,
        fs: int,
        loc: int,
        last_loc: int,
        peak: int,
        slope_diff: list,
        slope_heights: list) -> Tuple[list, list]:

    """

    :param c: the window that is considered as no two R peaks will be present
    :param sub_list: window that R peaks were not detected
    :param fs: sampling frequency
    :param loc: location of last R peak that was detected before the interval where no R peak was detected
    :param last_loc: location of the final R peak after the interval where no R peak was detected
    :param peak: height of the last peak before the interval where no R peak was detected
    :param slope_diff: slope difference values up to loc
    :param slope_heights: slope heights up to begin loc
    :return: tuple containing the lists of location and peak values
    """
    locations, peaks = rpeakdetection.new_r_peaks(sub_list, fs, c, loc, last_loc, peak, slope_diff, slope_heights)
    return locations, peaks



