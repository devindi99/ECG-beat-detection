import wfdb
import numpy as np
from beatdetection import rpeakdetection
from beatdetection import plot
from beatdetection import data_logging

remove_sym = ["+", "|", "~", "x", "]", "[", "U", " MISSB", "PSE", "TS", "T", "P", "M", "\""]


def read_annotations(
        name: int,
        path: str) -> tuple:
    """

    :param name: name of the record as an integer
    :param path: folder path where the record exist
    :return: annotation object
    """
    path = path + str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    annotations = wfdb.rdann(path, 'atr')
    heights = [signals[i][0] for i in range(len(signals))]
    return annotations, heights


def beat_pair(
        ref_annotations: np.ndarray,
        test_annotations: np.ndarray,
        fs: int = 360) -> np.ndarray:
    """

    :param ref_annotations: array of reference annotations
    :param test_annotations: array of the detected annotations
    :param fs: sampling frequency
    :return: paired list
    """
    ref_annotations = ref_annotations.flatten().reshape((len(ref_annotations), -1))
    test_annotations = test_annotations.flatten().reshape((len(test_annotations), -1))
    ref_ann_struct_init = np.hstack((ref_annotations, np.zeros((ref_annotations.shape[0], 5))))
    ref_ann_struct = ref_ann_struct_init
    test_ann_struct = np.hstack((test_annotations, np.zeros((test_annotations.shape[0], 5))))
    paired_list = []
    t_index = 0
    T_Index = 0
    T = ref_ann_struct[T_Index, 0]
    t = test_ann_struct[t_index, 0]
    threshold = (37 / 250) * fs
    if (abs(T - t) > threshold) and ((T - 0) < threshold):
        T_Index += 1
        T = ref_ann_struct[T_Index, 0]
    elif (abs(T - t) > threshold) and ((t - 0) < threshold):
        t_index += 1
        t = test_ann_struct[t_index, 0]

    while (t_index < test_ann_struct.shape[0] - 1) or (T_Index < ref_ann_struct.shape[0] - 1):
        if t_index >= test_ann_struct.shape[0] - 1:
            pair_element = [T_Index, -1]
            paired_list.append(pair_element)
            T_Index += 1
            t_index += 1
        elif T_Index >= ref_ann_struct.shape[0] - 1:
            pair_element = [-1, t_index]
            paired_list.append(pair_element)
            T_Index += 1
            t_index += 1
        elif t <= T:
            t_index += 1
            tdash = test_ann_struct[t_index, 0]
            if abs(T - t) <= abs(tdash - T) and abs(T - t) <= threshold:
                pair_element = [T_Index, t_index - 1]
                paired_list.append(pair_element)

                T_Index += 1
                T = ref_ann_struct[T_Index, 0]
                t = tdash

            elif abs(tdash - T) < abs(T - t) and abs(tdash - T) <= threshold:
                pair_element = [-1, t_index - 1]  # 0 = O, 1 = X, 2 = paired
                paired_list.append(pair_element)
                pair_element = [T_Index, t_index]
                paired_list.append(pair_element)
                T_Index += 1
                t_index += 1
                T = ref_ann_struct[T_Index, 0]
                if t_index <= test_ann_struct.shape[0] - 1:
                    t = test_ann_struct[t_index, 0]

            elif abs(tdash - T) > threshold and abs(T - t) > threshold:
                pair_element = [-1, t_index - 1]  # 0 = O, 1 = X, 2 = paired % t
                paired_list.append(pair_element)
                t = tdash
        else:
            T_Index += 1
            Tdash = ref_ann_struct[T_Index, 0]
            if abs(t - T) <= abs(Tdash - t) and abs(t - T) <= threshold:  # % T         T'
                pair_element = [T_Index - 1, t_index]
                paired_list.append(pair_element)

                t_index += 1
                t = test_ann_struct[t_index, 0]
                T = Tdash

            elif abs(Tdash - t) < abs(t - T) and abs(Tdash - t) <= threshold:
                pair_element = [T_Index - 1, -1]  # % 0 = O, 1 = X, 2 = paired
                paired_list.append(pair_element)
                pair_element = [T_Index, t_index]
                paired_list.append(pair_element)
                T_Index += 1
                t_index += 1
                T = ref_ann_struct[T_Index, 0]
                t = test_ann_struct[t_index, 0]

            elif (abs(Tdash - t) > threshold) and (abs(t - T) > threshold):
                pair_element = [T_Index - 1, -1]
                paired_list.append(pair_element)
                T = Tdash

    return np.asarray(paired_list)


def ref_annotate(
        record: int,
        path: str,
        count: int) -> tuple:

    """

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
                s = ann.sample[i]
                while ann.symbol[i] != "]":
                    e = ann.sample[i]
                    i += 1
            for m in range(s, e):
                a_fib.append(m)
            if ann.symbol[i] in remove_sym:
                continue
            else:
                if ann.sample[i] >= count:
                    ref_ann.append(heights[ann.sample[i]])
                    ref_loc.append(ann.sample[i])

        ref_locations = np.array(ref_loc)
        ref_annotations = np.array(ref_ann)

        return ref_locations, ref_annotations, a_fib

    except FileNotFoundError:
        print("File: ", record, "doesn't exist.")

        return [], [], []


def accuracy_check(
        ref_locations: list,
        ref_annotations: list,
        locations: list,
        peaks: list,
        fig: bool,
        show: bool) ->tuple:


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

