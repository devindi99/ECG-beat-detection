from beatdetection import rpeakdetection
from beatdetection import ptemplate
import matplotlib
from beatdetection import folderhandling
from beatdetection import beatpair
from beatdetection import recorrect
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


for record in range(203, 204):
    try:

        # output_dir = folder + str(record)
        # folderhandling.mkdir_p(output_dir)
        path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
        heights, fs = rpeakdetection.read_annotations(record, path)
        print(record)
        l = []
        p = []
        start = time.time()
        locations, peaks, count, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.375 * fs))
        k = len(locations)
        i = 0
        t = [i for i in range(len(heights))]
        plt.plot(t, heights)
        while i< k-1:

            state = recorrect.check_rr(locations[i], locations[i+1], round(0.6*fs))
            if state:

                l.append(locations[i])
                l.append(locations[i+1])
                p.append(peaks[i])
                p.append(peaks[i+1])
                pre_loc = locations[:i+1]
                post_loc = locations[i+1:]
                pre_peaks = peaks[:i+1]
                post_peaks = peaks[i+1:]

                add_locs, add_peaks = recorrect.check_peak(round(0.3 * fs), heights[locations[i]:locations[i+1]], fs, locations[i], locations[i+1], peaks[i], sdiffs[i-10:i], slope_heights[i-10:i])
                n = len(add_locs)
                # plt.scatter(add_locs, add_peaks, color="black")
                locations = pre_loc+add_locs+post_loc
                peaks = pre_peaks+add_peaks+post_peaks

                i += n
            i += 1

        plt.scatter(l, p, color="orange")

        end = time.time()
        ref_locations, ref_annotations, a_fib = beatpair.ref_annotate(record, path, fs)
        for m in range(len(a_fib)):
            if a_fib[m] in locations:
                o = locations.index(a_fib[m])
                locations.remove(a_fib[m])
                del peaks[o]

        # plt.scatter(locations, peaks, color="red", marker="x")
        # plt.show()
        TP, FP, FN, sensitivty, pp, DER = beatpair.accuracy_check(ref_locations, ref_annotations, locations, peaks,
                                                            True, True)
        print("TP: ", TP)
        print("FP: ", FP)
        print("FN: ", FN)

        if excel:
            ws1.append((record, len(locations), len(ref_locations), TP, FP,
                                       FN, sensitivty, pp, DER, end-start))

        # template = ptemplate.create_template([record], locations, heights)
        # t = [m for m in range(len(template))]
        # plt.figure()
        # plt.plot(t, template)
        # filename = str(record) + "template.png"
        # imagepath = output_dir + "/" + filename
        # plt.savefig(imagepath)
        # plt.close()
        # pwaves = ptemplate.p_waves(record, locations, heights)
        # for m in range(len(pwaves)):
        #     p = pwaves[m]
        #     distance = dtw.distance(p, template)
        #     if distance > 2.5:
        #         plt.figure()
        #         T = [m for m in range(len(p))]
        #         plt.plot(T, p)
        #         filename = str(record) + "distance " + str(distance) + ".png"
        #         imagepath = output_dir + "/" + filename
        #         plt.savefig(imagepath)
        #         plt.close()
    except FileNotFoundError:
        continue
if excel:
    wb.save(name)

