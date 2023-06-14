from beatdetection import rpeakdetection
from beatdetection import ptemplate
import matplotlib
from beatdetection import folderhandling
from beatdetection import beatpair
from beatdetection import recorrect
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
from openpyxl import Workbook
import time

folder = "Attempt3/"
folderhandling.mkdir_p(folder)
# check_accuracy = int(input("Accuracy checker ON/ OFF? "))
excel = int(input("Write accuracy values to a XL sheet? "))
if excel:
    wb = Workbook()
    name = input("Enter workbook name: ")
    sheet = input("Enter sheet name: ")
    ws1 = wb.create_sheet("Mysheet")
    ws1.title = sheet
#
records = ["118e_6", "118e00", "118e06", "118e12", "118e18", "118e24", "119e_6", "119e00", "119e06", "119e12", "119e18", "119e24"]

def calibration(heights, fs):
    l = []
    p = []

    locations, peaks, _, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.3 * fs), False, [], [], [], [])
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
    # plt.scatter(l, p, color="green")
    return locations, peaks, round(avg_RR), slope_heights, sdiffs


for record in records:
    try:

        # output_dir = folder + str(record)
        # folderhandling.mkdir_p(output_dir)
        path = 'D:\Semester 6\Internship\mit-bih-noise-stress-test-database-1.0.0/'
        heights, fs = rpeakdetection.read_annotations(record, path)
        t = [i for i in range(len(heights))]
        plt.plot(t, heights)
        print(record)
        l = []
        p = []

        start = time.time()
        cal_locations, cal_peaks, d, slope_heights, sdiffs = calibration(heights[:5 * 60 * fs + 1], fs)
        # plt.scatter(cal_locations, cal_peaks, color="blue")
        i = 0
        loc, pea, count, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.3 * fs), True,
                                                                               cal_locations, cal_peaks, slope_heights,
                                                                               sdiffs)
        # plt.scatter(loc, pea, color="blue")

        k = len(loc)

        while i < k - 1:

            state = recorrect.check_rr(loc[i], loc[i + 1], d)
            if state:
                l.append(loc[i])
                l.append(loc[i + 1])
                p.append(pea[i])
                p.append(pea[i + 1])

                pre_loc = loc[:i + 1]
                post_loc = loc[i + 1:]
                pre_peaks = pea[:i + 1]
                post_peaks = pea[i + 1:]

                add_locs, add_peaks = recorrect.check_peak(round(0.28 * fs), heights[loc[i]:loc[i + 1]], fs,
                                                           loc[i], loc[i + 1], pea[i], sdiffs[:i],
                                                           slope_heights[:i])
                n = len(add_locs)
                # plt.scatter(add_locs, add_peaks, color="black")
                loc = pre_loc + add_locs + post_loc
                pea = pre_peaks + add_peaks + post_peaks
                k += n
                i += n
            i += 1

        # plt.scatter(l, p, color="red", marker="o")
        loc = cal_locations + loc
        pea = cal_peaks + pea
        end = time.time()

        ref_locations, ref_annotations, a_fib = beatpair.ref_annotate(record, path)
        for m in range(len(a_fib)):
            if a_fib[m] in loc:
                o = loc.index(a_fib[m])
                loc.remove(a_fib[m])
                del pea[o]

        # plt.scatter(locations, peaks, color="red", marker="x")
        # plt.show()
        TP, FP, FN, sensitivty, pp, DER = beatpair.accuracy_check(ref_locations, ref_annotations, loc, pea,
                                                                 False, False)
        print("TP: ", TP)
        print("FP: ", FP)
        print("FN: ", FN)
        print(end - start)

        if excel:
            ws1.append((record, len(loc), len(ref_locations), TP, FP,
                        FN, sensitivty, pp, DER, end - start))
    except FileNotFoundError:
        continue
    if excel:
        wb.save(name)


