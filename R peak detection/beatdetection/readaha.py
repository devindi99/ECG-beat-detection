#  Copyright © Synergen Technology Labs LLC (https://synergentl.com/)
#  All Rights Reserved.
#  NOTICE: All information contained herein is, and remains
#  the property of Synergen Technology Labs LLC and its suppliers,
#  if any. The intellectual and technical concepts contained
#  herein are proprietary to Synergen Technology Labs LLC
#  and its suppliers and may be covered by United States and Foreign Patents or
#  patents in process and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material
#  is strictly forbidden unless prior written permission is obtained
#  from Synergen Technology Labs LLC

import os
from scipy.io import savemat
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Union, Optional, Tuple, List
from beatdetection import rpeakdetection
from beatdetection import beatpair
from openpyxl import Workbook
from beatdetection import recorrect
from beatdetection import filters

import time
from scipy import signal

# excel = int(input("Write accuracy values to a XL sheet? "))
# if excel:
#     wb = Workbook()
#     name = input("Enter workbook name: ")
#     sheet = input("Enter sheet name: ")
#     ws1 = wb.create_sheet("Mysheet")
#     ws1.title = sheet

AHA_records = ("1201", "1202", "1203", "1204", "1205",
               "1206", "1207", "1208", "1209", "1210",
               "2201", "2203", "2204", "2205",
               "2206", "2207", "2208", "2209", "2210",
               "3201", "3202", "3203", "3204", "3205",
               "3206", "3207", "3208", "3209", "3210",
               "4201", "4202", "4203", "4204", "4205",
               "4206", "4207", "4208", "4209", "4210",
               "5201", "5202", "5203", "5204", "5205",
               "5206", "5207", "5208", "5209", "5210",
               "6201", "6202", "6203", "6204", "6205",
               "6206", "6207", "6208", "6209", "6210",
               "7201", "7202", "7203", "7204", "7205",
               "7206", "7207", "7208", "7209", "7210",
               "8201", "8202", "8203", "8204", "8206",
               "8207", "8208", "8209", "8210"
               )  # total 78 records, 2 records excluded due to paced beats
AHA_sampled_freq = 250


def aha_ann_process(refpeak: npt.NDArray, reftype: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    The function will convert the peaks to five major beat classes
    """
    beatdic = {81: [12, 13, 16, 37, 38], 83: [4, 7, 8, 9, 11, 34, 35],
               86: [5, 10, 41], 70: [6], 78: [1, 2, 3], 85: [100],
               32: [32], 33: [33], 31: [31], 14: [14]}
    beatdicreverse = {}
    for key in beatdic.keys():
        for element in beatdic[key]:
            beatdicreverse[element] = key

    beatpos = refpeak  # later map the peaks to correct positions
    beattype = np.zeros((reftype.shape[0], 1))

    for beatidx in range(reftype.shape[0]):
        beat = reftype[beatidx, 0]
        if beat == 31:
            print('Ventricular flutter detected! This is not handled by the system')
        if beat in beatdicreverse.keys():
            beattype[beatidx] = beatdicreverse[beat]
        else:
            beattype[beatidx] = 255

    delidx_beat = np.where(np.logical_or(beattype == 255, beattype == 31) == True)
    # beatpos = np.delete(beatpos.astype(int), delidx_beat).reshape((-1, 1))
    # beattype = np.delete(beattype.astype(int), delidx_beat).reshape((-1, 1))

    return beatpos, beattype


def read_aharecord_reference(directory: str, record: str) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    The function will return the Lead 2 ECG samples in mV, and annotations in the order of [peak_locations,beat_annotations,rhythm_annotations]
    """

    signalfile = directory + record + '.mat'
    beatfile = directory + 'ann_' + record + '.mat'
    rpeakfile = directory + 'R_' + record + '.mat'

    if os.path.exists(signalfile) and os.path.exists(beatfile) and os.path.exists(rpeakfile):

        refdata = loadmat(signalfile)
        beatdata = loadmat(beatfile)
        peakdata = loadmat(rpeakfile)
        signal_II = refdata['M'][:, 0].reshape((-1, 1))
        signal_other = refdata['M'][:, 1].reshape((-1, 1))

        beattype_init = beatdata['ANNOT']
        peaks_init = peakdata['ATRTIME'] + 5

        peaks, beattype = aha_ann_process(peaks_init, beattype_init)
        beattype = beattype[np.where(np.logical_and(0 <= peaks, peaks < len(signal_II)))[0]]
        peaks = peaks[np.where(np.logical_and(0 <= peaks, peaks < len(signal_II)))[0]]
        return signal_II, np.hstack((peaks, beattype, np.zeros((peaks.shape[0], 1)))).astype(int), signal_other
    else:
        raise IOError(f'Record {record} is not found in the directory. VERIFY the file-directory path.')


def plot_peaks(signal: npt.NDArray, annotation: npt.NDArray, record: str) -> tuple:
    """
    Function will plot the ECG signals with peaks in a scatter plot for verification or debug
    """
    if annotation.ndim == 2:
        peaks = annotation[:, 0]
        # plt.figure(1)
        # plt.title(f'AHA record {record}: ECG signal with peaks')
        # plt.plot(signal)
        # plt.scatter(peaks, signal[peaks], c='b')
        # plt.show()
        return peaks, signal[peaks]
    else:
        ValueError('Check input array dimensions ')


# def calibration(heights, fs):
#     l = []
#     p = []
#
#     locations, peaks, count, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.375 * fs), False, [], [], [], [])
#     k = len(locations)
#     i = 0
#
#     while i < k - 1:
#         state = recorrect.check_rr(locations[i], locations[i + 1], round(0.7 * fs))
#         if state:
#             l.append(locations[i])
#             l.append(locations[i + 1])
#             p.append(peaks[i])
#             p.append(peaks[i + 1])
#             pre_loc = locations[:i + 1]
#             post_loc = locations[i + 1:]
#             pre_peaks = peaks[:i + 1]
#             post_peaks = peaks[i + 1:]
#
#             add_locs, add_peaks = recorrect.check_peak(round(0.25 * fs), heights[locations[i]:locations[i + 1]], fs,
#                                                        locations[i], locations[i + 1], peaks[i], sdiffs[:i++1],
#                                                        slope_heights[:i+1])
#
#             n = len(add_locs)
#
#             locations = pre_loc + add_locs + post_loc
#             peaks = pre_peaks + add_peaks + post_peaks
#             k += n
#             i += n
#         i += 1
#
#     RR = []
#     for m in range(len(locations)-1):
#         RR.append(locations[m+1] - locations[m])
#     avg_RR = np.average(RR)
#     # plt.scatter(l, p, color="green")
#     return locations, peaks, round(avg_RR), slope_heights, sdiffs


def main(file_dir: str, file_list: Optional[Union[Tuple[str], List[str]]] = None) -> None:
    global AHA_records, AHA_sampled_freq
    if file_list is None:
        file_list = AHA_records
    for j in range(len(file_list)):
        ecg, ann, signal_other = read_aharecord_reference(file_dir, file_list[j])
        # t = [i for i in range(len(ecg))]
        # plt.plot(t, ecg)
        plt.title(f'AHA record {file_list[j]}: ECG signal with peaks')
        # plt.show()

        signal_other = filters.Low_pass(signal_other)
        signal_other = filters.iir(signal_other, 2000)
        ecg = filters.Low_pass(ecg)
        ecg = filters.iir(ecg, 2000)
        t1 = [i for i in range(len(signal_other))]
        t = [i for i in range(len(ecg))]
        plt.plot(t, ecg)
        plt.plot(t1, signal_other)
        plt.show()
    #     ref_locations, ref_annotations = plot_peaks(ecg, ann, file_list[j])
    #     vfib=[]
    #     for i in range(len(ann)):
    #         if ann[i][1] == 32:
    #             try:
    #                 for m in range(ann[i][0],ann[i+1][0]):
    #                     vfib.append(m)
    #                 # d = [m for m in range(ann[i][0],ann[i+1][0])]
    #             except IndexError:
    #                 for m in range(ann[i][0], len(ecg)):
    #                     vfib.append(m)
    #             # ecg = np.delete(ecg, d)
    #     # ecg = ecg[:ref_locations[-1]+40]
    #     start = time.time()
    #     cal_locations, cal_peaks, d, slope_heights, sdiffs = calibration(ecg[:5 * 60 * AHA_sampled_freq + 1], AHA_sampled_freq)
    #     loc, pea, count, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(ecg, AHA_sampled_freq, round(0.375 *  AHA_sampled_freq ), True,
    #                                                                            cal_locations, cal_peaks, slope_heights,
    #                                                                            sdiffs)
    #     # plt.scatter(loc, pea, color="red")
    #     # locations, peaks, count, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(ecg, 250, round(0.375 * AHA_sampled_freq))
    #     print(file_list[j])
    #
    #     k = len(loc)
    #     i = 0
    #     l = []
    #     p = []
    #
    #
    #     while i < k - 1:
    #         state = recorrect.check_rr(loc[i], loc[i + 1], d)
    #         if state:
    #             l.append(loc[i])
    #             l.append(loc[i + 1])
    #             p.append(pea[i])
    #             p.append(pea[i + 1])
    #
    #             pre_loc = loc[:i + 1]
    #             post_loc = loc[i + 1:]
    #             pre_peaks = pea[:i + 1]
    #             post_peaks = pea[i + 1:]
    #
    #             add_locs, add_peaks = recorrect.check_peak(round(0.25 * AHA_sampled_freq), ecg[loc[i]:loc[i + 1]],AHA_sampled_freq,
    #                                                        loc[i], loc[i + 1], pea[i], sdiffs[:i],
    #                                                        slope_heights[:i])
    #             n = len(add_locs)
    #             # plt.scatter(add_locs, add_peaks, color="red")
    #             loc = pre_loc + add_locs + post_loc
    #             pea = pre_peaks + add_peaks + post_peaks
    #             k += n
    #             i += n
    #         i += 1
    #     # loc = cal_locations + loc
    #     # pea = cal_peaks + pea
    #     plt.scatter(l, p, color="blue")
    #     end = time.time()
    #     # mydata = np.array([(1, 1.0), (2, 2.0)], dtype=[('foo', 'i'), ('bar', 'f')])
    #     # filename = file_list[j] + ".mat"
    #     # savemat(filename, {'testann': np.array(locations), "refann": np.array(ref_locations)})
    #     # data = loadmat(filename)
    #     # print(data["testann"])
    #     # print(data["refann"])
    #     for m in range(len(vfib)):
    #         if vfib[m] in loc:
    #             o = loc.index(vfib[m])
    #             loc.remove(vfib[m])
    #             del pea[o]
    #     TP, FP, FN, sensitivty, pp, DER = beatpair.accuracy_check(ref_locations, ref_annotations, loc, pea,
    #                                                             True, True)
    #     print("TP: ", TP)
    #     print("FP: ", FP)
    #     print("FN: ", FN)
    #     if excel:
    #         ws1.append((file_list[j], len(loc), len(ref_locations), TP, FP,
    #                                    FN, sensitivty, pp, DER, end-start))
    #     # plt.scatter(locations, peaks, c="g", marker="x")
    #     # plt.show()
    # if excel:
    #     wb.save(name)


if __name__ == '__main__':
    # check_file_list = list(AHA_records)
    check_file_list = ["5203"]
    file_loc = 'D:/Semester 6/Internship/AHA_data/'
    main(file_loc, file_list=check_file_list)
