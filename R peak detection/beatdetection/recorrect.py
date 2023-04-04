from beatdetection import rpeakdetection
import numpy as np


def check_rr(sample1, sample2, d):
    if (sample2 - sample1 > d):
        return True
    else:
        return False


def check_peak(c, sub_list, fs, loc, end, peak, sd, sl):
    locations, peaks = rpeakdetection.new_r_peaks(sub_list, fs, c, loc, end, peak, sd, sl)
    # print(locations)
    return locations, peaks



