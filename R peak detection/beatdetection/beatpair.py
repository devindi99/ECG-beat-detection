"""
Module for reading from annotation file using WFDB tool box, and calculating the number of True Positives, False
Positives and False negatives.
"""

import wfdb
import numpy as np
from typing import Tuple, List

from beatdetection import rpeakdetection
from beatdetection import plot
from scipy import signal

remove_sym = ["+", "|", "~", "x", "]", "[", "U", " MISSB", "PSE", "TS", "T", "P", "M", "\"", "!"]


def read_annotations(
        name: int,
        path: str) -> tuple:
    """
    This function is used to read atr files.

    :param name: name of the record as an integer
    :param path: folder path where the record exist
    :return: annotation object
    """
    path = path + str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    annotations = wfdb.rdann(path, 'atr')
    heights = [signals[i][0] for i in range(len(signals))]
    resampled = signal.resample_poly(heights, 250, 360)
    return annotations, resampled


def beat_pair(refannot, testannot, fs=360):
    refannot = refannot.flatten().reshape((np.size(refannot), -1))
    testannot = testannot.flatten().reshape((np.size(testannot),-1))
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
                if TIndex >= refannstruct.shape[0] - 1:
                    continue
                T = refannstruct[TIndex, 0]
                t = tdash

            elif abs(tdash - T) < abs(T - t) and abs(tdash - T) <= threshold:
                pairelement = [-1, tindex - 1]  # 0 = O, 1 = X, 2 = paired
                pairedlist.append(pairelement)
                pairelement = [TIndex, tindex]
                pairedlist.append(pairelement)
                TIndex += 1
                tindex += 1
                if TIndex >= refannstruct.shape[0] - 1:
                    continue
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
                if tindex >= testannstruct.shape[0] - 1:
                    continue
                t = testannstruct[tindex, 0]
                T = Tdash

            elif abs(Tdash - t) < abs(t - T) and abs(Tdash - t) <= threshold:
                pairelement = [TIndex - 1, -1]  # % 0 = O, 1 = X, 2 = paired
                pairedlist.append(pairelement)
                pairelement = [TIndex, tindex]
                pairedlist.append(pairelement)
                TIndex += 1
                tindex += 1
                if TIndex >= refannstruct.shape[0] - 1:
                    continue
                if tindex >= testannstruct.shape[0] - 1:
                    continue
                T = refannstruct[TIndex, 0]
                t = testannstruct[tindex, 0]

            elif (abs(Tdash - t) > threshold) and (abs(t - T) > threshold):
                pairelement = [TIndex - 1, -1]
                pairedlist.append(pairelement)
                T = Tdash

    return np.asarray(pairedlist)


def ref_annotate(
        record: int,
        path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:

    """
    Returns reference beat locations and heights, and afib annotations

    :param record: name of the record as an integer
    :param path: folder path where the record exist
    :return: reference beat locations and heights, afib indexes
    """
    try:
        ann, heights = read_annotations(record, path)
        ref_ann = []
        ref_loc = []
        a_fib = []
        for i in range(len(ann.sample)):
            s = 0
            e = 0
            if ann.symbol[i] == "[":
                s = round(ann.sample[i]*250/360)
                while ann.symbol[i] != "]":
                    e = round(ann.sample[i]*250/360)
                    i += 1
            for m in range(s, e):
                a_fib.append(m)
            if ann.symbol[i] in remove_sym:
                continue
            else:
                # print(round(ann.sample[i]*250/360))
                ref_ann.append(heights[round(ann.sample[i]*250/360)])
                ref_loc.append(round(ann.sample[i]*250/360))

        ref_locations = np.array(ref_loc)
        ref_annotations = np.array(ref_ann)

        return ref_locations, ref_annotations, a_fib

    except FileNotFoundError:
        print("File: ", record, "doesn't exist.")


def accuracy_check(
        ref_locations: np.ndarray,
        ref_annotations: np.ndarray,
        locations: list,
        peaks: list,
        fig: bool,
        show: bool) -> Tuple[int, int, int, float, float, float]:
    """
    Calculate the number of True Positives, False negatives, False positives, sensitivity, positive predictivity and
    error rate

    :param ref_locations: Reference peak locations
    :param ref_annotations: Reference peak heights
    :param locations: Detected peak locations
    :param peaks: Detected peak heights
    :param fig: True -> ECG signals are plotted, False -> otherwise
    :param show: Display signal and annotations as per required by the user
    :return: number of True Positives, False negatives, False positives, sensitivity, positive predictivity and
    error rate
    """
    TP = 0
    FP = 0
    FN = 0
    TP_list = []
    FP_list = []
    FN_list = []
    TP_list_loc = []
    FP_list_loc = []
    FN_list_loc = []

    paired_list = beat_pair(np.array(ref_locations), np.array(locations))
    for i in range(len(paired_list)):
        if paired_list[i][0] == -1:
            FP += 1
            FP_list.append(peaks[paired_list[i][1]])
            FP_list_loc.append(locations[paired_list[i][1]])
        elif paired_list[i][1] == -1:
            FN += 1
            FN_list.append(ref_annotations[paired_list[i][0]])
            FN_list_loc.append(ref_locations[paired_list[i][0]])
        else:
            TP += 1
            TP_list.append(peaks[paired_list[i][1]])
            TP_list_loc.append(locations[paired_list[i][1]])

    sensitivty = TP * 100 / (TP + FN)
    pp = TP * 100 / (TP + FP)
    DER = (FP + FN) / len(ref_locations)

    if fig:
        plot.plotter([[TP_list, TP_list_loc], [FP_list, FP_list_loc], [FN_list, FN_list_loc]], True, show)

    return TP, FP, FN, sensitivty, pp, DER
