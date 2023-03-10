
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import Workbook
from BaselineRemoval import BaselineRemoval
import time
workbook = Workbook()


wb = openpyxl.Workbook()

sheet = wb.active
NST_DB=["118e_6","118e00","118e06","118e12","118e18","118e24","119e_6","119e00","119e06","119e12","119e18","119e24"]


def beatpair (refannot: np.ndarray, testannot: np.ndarray, fs: int = 360) -> np.ndarray:
    """

    :param refannot: array of referrance annotations
    :param testannot: array of the detected annotations
    :param fs: sampling frequency
    :return: paired list
    """
    refannot = refannot.flatten().reshape((len(refannot), -1))
    testannot = testannot.flatten().reshape((len(testannot),-1))
    refannstructinit = np.hstack((refannot, np.zeros((refannot.shape[0], 5))))
    refannstruct = refannstructinit
    testannstruct = np.hstack((testannot, np.zeros((testannot.shape[0], 5))))
    pairedlist = []
    tindex = 0
    TIndex = 0
    T = refannstruct[TIndex, 0]
    t = testannstruct[tindex, 0]
    threshold = (37 / 250) * fs
    if (abs(T - t) > threshold) and ((T - 0) < threshold):
        TIndex += 1
        T = refannstruct[TIndex, 0]
    elif (abs(T - t) > threshold) and ((t - 0) < threshold):
        tindex += 1
        t = testannstruct[tindex, 0]

    while (tindex < testannstruct.shape[0] - 1) or (TIndex < refannstruct.shape[0] - 1):
        if tindex >= testannstruct.shape[0] - 1:
            pairelement = [TIndex, -1]
            pairedlist.append(pairelement)
            TIndex += 1
            tindex += 1
        elif TIndex >= refannstruct.shape[0] - 1:
            pairelement = [-1, tindex]
            pairedlist.append(pairelement)
            TIndex += 1
            tindex += 1
        ## step (b)
        elif t <= T:
            tindex += 1
            tdash = testannstruct[tindex, 0]
            if abs(T - t) <= abs(tdash - T) and abs(T - t) <= threshold:
                pairelement = [TIndex, tindex - 1]
                pairedlist.append(pairelement)

                TIndex += 1
                T = refannstruct[TIndex, 0]
                t = tdash

            elif abs(tdash - T) < abs(T - t) and abs(tdash - T) <= threshold:
                pairelement = [-1, tindex - 1]  # 0 = O, 1 = X, 2 = paired
                pairedlist.append(pairelement)
                pairelement = [TIndex, tindex]
                pairedlist.append(pairelement)
                TIndex += 1
                tindex += 1
                T = refannstruct[TIndex, 0]
                if tindex <= testannstruct.shape[0] - 1:
                    t = testannstruct[tindex, 0]

            elif abs(tdash - T) > threshold and abs(T - t) > threshold:
                pairelement = [-1, tindex - 1]  # 0 = O, 1 = X, 2 = paired % t
                pairedlist.append(pairelement)
                t = tdash
        else:
            TIndex += 1
            Tdash = refannstruct[TIndex, 0]
            if abs(t - T) <= abs(Tdash - t) and abs(t - T) <= threshold:  # % T         T'
                pairelement = [TIndex - 1, tindex]
                pairedlist.append(pairelement)

                tindex += 1
                t = testannstruct[tindex, 0]
                T = Tdash

            elif abs(Tdash - t) < abs(t - T) and abs(Tdash - t) <= threshold:
                pairelement = [TIndex - 1, -1]  # % 0 = O, 1 = X, 2 = paired
                pairedlist.append(pairelement)
                pairelement = [TIndex, tindex]
                pairedlist.append(pairelement)
                TIndex += 1
                tindex += 1
                T = refannstruct[TIndex, 0]
                t = testannstruct[tindex, 0]

            elif (abs(Tdash - t) > threshold) and (abs(t - T) > threshold):
                pairelement = [TIndex - 1, -1]
                pairedlist.append(pairelement)
                T = Tdash

    return np.asarray(pairedlist)


def readannotations(name: int):
    """

    :param name: name of the record
    :return:
    """
    path = 'D:\Semester 6\Internship\mit-bih-arrhythmia-database-1.0.0/' + str(name)
    # path = 'D:\Semester 6\Internship\mit-bih-noise-stress-test-database-1.0.0/' + str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    annotations = wfdb.rdann(path, 'atr')
    heights = [signals[i][0] for i in range(len(signals))]
    # resampled = signal.resample_poly(heights, 250, 360)
    time_re = [i for i in range(0, len(heights))]
    # baseObj = BaselineRemoval(heights)
    # Zhangfit_output = baseObj.ZhangFit()
    # plt.figure()
    # plt.plot(time_re, heights)
    return annotations, signals, heights, fields["fs"]
#functions for checking the first criterion


def max_min_slopes(n: int, samples: list, fs: int):
    """

    :param n: number of current sample
    :param samples:  sample values of ECG signal
    :param fs: sampling frequency
    :return: max and min slopes on the either sides of the sample
    """
    a = round(0.027*fs)
    b = round(0.063*fs)
    Lslopes = []
    Rslopes = []
    Lheights = []
    Rheights = []
    for k in range(a, b+1):
        if n+k >= len(samples) or n-k < 0:
            break
        Lslopes.append((samples[n]-samples[n-k])/k)
        Rslopes.append((samples[n]-samples[n+k])/(-1*k))
        Lheights.append(np.absolute(samples[n] - samples[n - k]))
        Rheights.append(np.absolute(samples[n] - samples[n + k]))
    return max(Rslopes), min(Rslopes), max(Lslopes), min(Lslopes),max(Rheights), max(Lheights)


def max_slope_difference(max_R: int, min_R: int, max_L: int, min_L:  int, max_Lheight: int, max_Rheight: int):
    """

    :param max_R: maximum slope to the right side of the considered sample
    :param min_R: minimum slope to the right side of the considered sample
    :param max_L: maximum slope to the left side of the considered sample
    :param min_L: minimum slope to the left side of the considered sample
    :param max_Lheight: slope height of the left side
    :param max_Rheight: slope height of the right side
    :return: maximum slope difference value, slope height
    """
    if max_L - min_R > max_R - min_L:
        return max_L - min_R , max_Lheight
    else:
        return max_R - min_L, max_Rheight


def teeta_diff(peaks: list,fs: int)-> float:
    """

    :param peaks: peaks that have been detected so far
    :param fs: sampling frequency
    :return: threshold value
    """
    s_avg = np.average(np.absolute(peaks[-8:]))

    if s_avg > 20.480 / fs:
        return 7.680 / fs
    elif 2.800 / fs < s_avg < 20.480 / fs:
        return 4.352 / fs
    else:
        return 3.840 / fs


def first_criterion(teeta_diff: float, Sdiff_max: float) ->bool:
    """

    :param teeta_diff: threshold value
    :param Sdiff_max: maximum difference between slopes on either sides of the sample
    :return: True-> sample follow first criterion, False-> otherwise
    """
    return Sdiff_max > teeta_diff


def S_min(max_R: int, min_R: int, max_L: int, min_L: int):
    """

    :param max_R: maximum slope to the right side of the considered sample
    :param min_R: minimum slope to the right side of the considered sample
    :param max_L: maximum slope to the left side of the considered sample
    :param min_L: minimum slope to the left side of the considered sample
    :return: minimum of slopes, True-> local min/max False->otherwise
    """
    if max_L - min_R > max_R - min_L:
        Smin = min (np.abs(max_L), np.abs(min_R))
        state = (np.sign(max_L) == -1 * np.sign(min_R))

    else:
        Smin = min (np.abs(max_R), np.abs(min_L))
        state = (np.sign(max_R) == -1 * np.sign(min_L))
    return Smin, state


def second_criterion(Smin: int, state: bool, fs: int) -> bool:
    """

    :param Smin:
    :param state:
    :param fs:
    :return:
    """

    if Smin > 1.536/fs and state:
        return True
    else:
        return False



def third_criterion(cur_height: float, peaks: list)->bool:
    """

    :param cur_height: float
    :param peaks: list
    :return: bool
    """
    try:
        h_avg = np.average(np.absolute(peaks[-8:]))
    except IndexError:
        h_avg = np.average(np.absolute(peaks[:]))
    return np.abs(cur_height) > h_avg * 0.4


remove_sym = ["+","|", "~", "x", "]", "[", "U", " MISSB", "PSE", "TS", "T", "P", "M", "\"", "!"]


for record in range(100,235):
    try:
        TP = 0
        FP = 0
        FN = 0
        TP_list = []
        FP_list = []
        FN_list = []
        TP_list_loc = []
        FP_list_loc = []
        FN_list_loc = []
        print(record)
        annotations, signals, heights, fs = readannotations(record)
        Refannotations = []
        Reflocations = []
        AFIB= {}
        for i in range(len(annotations.sample)):
            s = 0
            e = 0
            if annotations.symbol[i] == "[":
                s = annotations.sample[i]
                while annotations.symbol[i] != "]":
                    e = annotations.sample[i]
                    i += 1
                AFIB[str(s)] = str(e)
            if annotations.symbol[i] in remove_sym:
                continue
            else:
                Refannotations.append(heights[annotations.sample[i]])
                Reflocations.append(annotations.sample[i])

        peaks = [Refannotations[0]]
        locations = [Reflocations[0]]
        slope_heights = []
        Sdiffs = []
        RR_Intervals=[]
        a = round(0.027 * fs)
        b = round(0.063 * fs)
        c = round(0.3125 * fs)
        for i in range(1, 8):
            peaks.append(Refannotations[i])
            locations.append(Reflocations[i])
            RR_Intervals.append(locations[-1] - locations[-2])
        start = time.time()
        for i in range(b, 3 * fs):
            max_R, min_R, max_L, min_L, max_Rheight, max_Lheight = max_min_slopes(i, heights, fs)
            Sdiff_max, max_height = max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight)
            slope_heights.append(max_height)
            Sdiffs.append(Sdiff_max)

        # callibration_time = 3 * 60 * fs
        for i in range(b, len(heights) + 1 - b):
            if i in AFIB:
                continue
            max_R, min_R, max_L, min_L, max_Rheight, max_Lheight = max_min_slopes(i, heights, fs)
            Sdiff_max, max_height = max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight)
            teeta = teeta_diff(Sdiffs, fs)
            Smin, state = S_min(max_R, min_R, max_L, min_L)
            QRS_complex = first_criterion(teeta, Sdiff_max)and second_criterion(Smin, state, fs)and third_criterion(
                max_height, slope_heights)

            if QRS_complex:
                element = max(np.absolute(heights[i - a:i + a + 1]))
                loc = np.where(np.absolute(heights[i - a:i + a + 1]) == element)
                loc = loc[0][0] + i - a
                if loc - c > locations[-1]:
                    locations.append(loc)
                    # print(loc)
                    peaks.append(heights[loc])
                    slope_heights.append(max_height)
                    Sdiffs.append(Sdiff_max)
                else:
                    if Sdiff_max > Sdiffs[-1]:
                        peaks[-1] = heights[loc]
                        locations[-1] = loc
                        # print("   xas",loc)
                        slope_heights[-1] = max_height
                        Sdiffs[-1] = Sdiff_max
        # RR = []
        # c = 0
        # n = len(locations)
        # for m in range(n-1):
        #     RR.append(locations[m + 1] - locations[m])
        #     #     c+=(locations[m+1]-locations[m])/20
        #     # c-=100
        # c = np.average(RR) - 110
        # print(c)
        # i = callibration_time
        # i=locations[-1]
        # while i < len(heights) + 1 - b:
        #     if i % callibration_time == 0:
        #         c = 0
        #         RR = []
        #         n = len(locations)
        #
        #         for m in range(n -10, n - 1):
        #             RR.append(locations[m + 1] - locations[m])
        #         c = np.average(RR) - 110
        #         print(c)
        #
        #     if str(i) in AFIB.keys():
        #         i = int(AFIB[str(i)])
        #         continue
        #
        #     max_R, min_R, max_L, min_L, max_Rheight, max_Lheight = max_min_slopes(i, heights, fs)
        #     Sdiff_max, max_height = max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight)
        #     teeta = teeta_diff(Sdiffs, fs)
        #     Smin, state = S_min(max_R, min_R, max_L, min_L)
        #     QRS_complex = first_criterion(teeta, Sdiff_max) and second_criterion(Smin, state, fs) and third_criterion(
        #         max_height, slope_heights)
        #     if QRS_complex:
        #         # print("wtrses")
        #         element = max(np.absolute(heights[i - a:i + a + 1]))
        #         loc = np.where(np.absolute(heights[i - a:i + a + 1]) == element)
        #         loc = loc[0][0] + i - a
        #         # locations.append(loc)
        #         # # print(loc)
        #         # peaks.append(heights[loc])
        #         # slope_heights.append(max_height)
        #         # Sdiffs.append(Sdiff_max)
        #         # i = locations[-1]
        #         # print(i)
        #         if loc - c > locations[-1]:
        #             locations.append(loc)
        #             # print(loc)
        #             peaks.append(heights[loc])
        #             slope_heights.append(max_height)
        #             Sdiffs.append(Sdiff_max)
        #             i = locations[-1]
        #
        #         else:
        #             if Sdiff_max > Sdiffs[-1]:
        #                 peaks[-1] = heights[loc]
        #                 locations[-1] = loc
        #                 # print("   xas",loc)
        #                 slope_heights[-1] = max_height
        #                 Sdiffs[-1] = Sdiff_max
        #                 i = locations[-1]
        #     i += 1 + round(0.02*fs)

        # Accurate_ann_loc = []
        # Accurate_ann_height = []
        # Incorrect_ann_loc = []
        # Incorrect_ann_height = []
        # MIT_ann = []
        # MIT_loc = []
        #
        # correct = 0
        # for i in range(len(locations)):
        #     if locations[i] in Reflocations:
        #         Accurate_ann_loc.append(locations[i])
        #         Accurate_ann_height.append(peaks[i])
        #         correct += 1
        #     else:
        #         Incorrect_ann_loc.append(locations[i])
        #         Incorrect_ann_height.append(peaks[i])
        #
        # # plt.scatter(locations, peaks, color="red", marker='x')
        # plt.scatter(Accurate_ann_loc, Accurate_ann_height, color="red", marker='x')
        # plt.scatter(Incorrect_ann_loc, Incorrect_ann_height, color="black", marker='o')
        # # plt.scatter(Reflocations,Refannotations, color="green", marker='x')
        # for i in range(len(Reflocations)):
        #     if Reflocations[i] in Accurate_ann_loc:
        #         continue
        #     else:
        #         MIT_ann.append(heights[Reflocations[i]])
        #         MIT_loc.append(Reflocations[i])
        #
        # plt.scatter(MIT_loc, MIT_ann, color="green", marker='x')
        # plt.show()
        end=time.time()
        locations = np.array(locations[7:])
        peaks = np.array(peaks[7:])
        Reflocations = np.array(Reflocations)
        Refannotations = np.array(Refannotations)
        pairedlist = beatpair(Reflocations, locations)
        for i in range(len(pairedlist)):
            if pairedlist[i][0] == -1:
                FP += 1
                FP_list.append(peaks[pairedlist[i][1]])
                FP_list_loc.append(locations[pairedlist[i][1]])
            elif pairedlist[i][1] == -1:
                FN += 1
                FN_list.append(Refannotations[pairedlist[i][0]])
                FN_list_loc.append(Reflocations[pairedlist[i][0]])
            else:
                TP += 1
                TP_list.append(peaks[pairedlist[i][1]])
                TP_list_loc.append(locations[pairedlist[i][1]])

        sensitivty = TP * 100 / (TP + FN)
        pp = TP * 100 / (TP + FP)
        DER = (FP + FN) / len(Reflocations)
        # plt.scatter(TP_list_loc, TP_list, color="red", marker="x")
        # plt.scatter(FN_list_loc, FN_list, color="green", marker="x")
        # plt.scatter(FP_list_loc, FP_list, color="black", marker="x")
        print("TP: ", TP)
        print("FP: ", FP)
        print("FN: ", FN)
        # plt.show()
        data = (record, len(locations), len(Reflocations), len(Reflocations) - len(locations),TP,FP,FN, sensitivty, pp, DER,end-start)
        sheet.append(data)
    except FileNotFoundError:
        continue
wb.save("1.536.xlsx")