import wfdb
import numpy as np
from beatdetection import rpeakdetection

remove_sym = ["+", "|", "~", "x", "]", "[", "U", " MISSB", "PSE", "TS", "T", "P", "M", "\"", "!"]


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


def test_annotate(
        record: int,
        path: str,
        n: int) -> tuple:

    """

    :param record: name of the record as an integer
    :param path: folder path where the record exist
    :param n: time (seconds) upto which the ECG data will be analysed
    :return: deatected beat locations and heights, reference beat locations and heights, paired list
    """
    try:
        annotations, heights = read_annotations(record, path)
        ref_annotations = []
        ref_locations = []

        for i in range(len(annotations.sample)):
            if annotations.symbol[i] in remove_sym:
                continue
            else:
                ref_annotations.append(heights[annotations.sample[i]])
                ref_locations.append(annotations.sample[i])

        locations, peaks = rpeakdetection.locate_r_peaks(record, path, n)
        locations = np.array(locations)
        peaks = np.array(peaks)
        ref_locations = np.array(ref_locations)
        ref_annotations = np.array(ref_annotations)
        paired_list = beat_pair(ref_locations, locations)

        return locations, peaks, ref_locations, ref_annotations, paired_list

    except FileNotFoundError:
        print("File: ", record, "doesn't exist.")

        return [], [], [], [], []




