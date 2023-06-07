from beatdetection import rpeakdetection
import numpy as np
from beatdetection import beatpair
from beatdetection import recorrect
from openpyxl import Workbook
import matplotlib.pyplot as plt
import time
import scipy


excel = int(input("Write accuracy values to a XL sheet? "))
if excel:
    wb = Workbook()
    name = input("Enter workbook name: ")
    sheet = input("Enter sheet name: ")
    ws1 = wb.create_sheet("Mysheet")
    ws1.title = sheet


def calibration(
        heights: list,
        fs: int) -> tuple:

    """

    :param heights: ECG signal values up to the first five minutes
    :param fs: sampling frequency
    :return: tuple(R peak locations: list, R peak heights: list, average RR interval: float, slope heights: list,
    slope differences: list)
    """
    locations, peaks, _, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.5 * fs), False, [], [], [], [])
    k = len(locations)
    i = 0

    while i < k - 1:
        state = recorrect.check_rr(locations[i], locations[i + 1], round(0.7 * fs))
        if state:
            pre_loc = locations[:i + 1]
            post_loc = locations[i + 1:]
            pre_peaks = peaks[:i + 1]
            post_peaks = peaks[i + 1:]

            add_locs, add_peaks = recorrect.check_peak(round(0.28 * fs), heights[locations[i]:locations[i + 1]], fs,
                                                       locations[i], locations[i + 1], peaks[i], sdiffs[:i++1],
                                                       slope_heights[:i+1])

            n = len(add_locs)

            locations = pre_loc + add_locs + post_loc
            peaks = pre_peaks + add_peaks + post_peaks
            k += n
            i += n
        i += 1

    RR = []
    for m in range(len(locations)-1):
        RR.append(locations[m+1] - locations[m])
    avg_RR = np.average(RR)
    return locations, peaks, round(avg_RR), slope_heights, sdiffs


for record in range(113, 235):

    try:
        remove = [102, 104, 107, 217]
        if record in remove:
            continue
        path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
        heights, fs = rpeakdetection.read_annotations(record, path)



        t = [i for i in range(len(heights))]
        plt.plot(t, heights)
        print(record)

        start = time.time()
        cal_locations, cal_peaks, d, slope_heights, sdiffs = calibration(heights[:5*60*fs+1], fs)
        print(d/fs)
        i = 0
        loc, pea, _, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.5 * fs), True,
                                                                               cal_locations, cal_peaks, slope_heights, sdiffs)

        k = len(loc)

        while i < k-1:
            l = []
            p = []
            state = recorrect.check_rr(loc[i], loc[i+1], round(0.7 * fs))
            if state:
                pre_loc = loc[:i+1]
                post_loc = loc[i+1:]
                pre_peaks = pea[:i+1]
                post_peaks = pea[i+1:]

                for m in range(loc[i], loc[i+1]+1):
                    l.append(m)
                    p.append(heights[m])
                plt.plot(l, p, color="red")
                add_locs, add_peaks = recorrect.check_peak(round(0.28 * fs), heights[loc[i]:loc[i+1]], fs,
                                                           loc[i], loc[i+1], pea[i], sdiffs[:i],
                                                           slope_heights[:i])
                n = len(add_locs)
                loc = pre_loc+add_locs+post_loc
                pea = pre_peaks+add_peaks+post_peaks
                k += n
                i += n
            i += 1

        loc = cal_locations + loc
        pea = cal_peaks + pea
        end = time.time()

        ref_locations, ref_annotations, a_fib = beatpair.ref_annotate(record, path)
        for m in range(len(a_fib)):
            if a_fib[m] in loc:
                o = loc.index(a_fib[m])
                loc.remove(a_fib[m])
                del pea[o]

        TP, FP, FN, sensitivty, pp, DER = beatpair.accuracy_check(ref_locations, ref_annotations, loc,  pea,
                                                          True, True)
        print("TP: ", TP)
        print("FP: ", FP)
        print("FN: ", FN)
        print(end-start)

        if excel:
            ws1.append((record, len(loc), len(ref_locations), TP, FP,
                                       FN, sensitivty, pp, DER, end-start))
    except FileNotFoundError:
        continue
if excel:
    wb.save(name)
