import matplotlib.pyplot as plt
from beatdetection import rpeakdetection
from beatdetection import beatpair
import time
from openpyxl import Workbook

# folder = "Attempt3/"
# folderhandling.mkdir_p(folder)
# check_accuracy = int(input("Accuracy checker ON/ OFF? "))
excel = int(input("Write accuracy values to a XL sheet? "))
if excel:
    wb = Workbook()
    name = input("Enter workbook name: ")
    sheet = input("Enter sheet name: ")
    ws1 = wb.create_sheet("Mysheet")
    ws1.title = sheet
for record in range(105, 235):

    try:
        remove = [102, 104, 107, 217]
        if record in remove:
            continue
        start = time.time()
        path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
        heights, fs = rpeakdetection.read_annotations(record, path)
        t = [i for i in range(len(heights))]
        # plt.plot(t, heights)

        fir = [0,0,0,0,0,0]
        for i in range(6, len(heights)):
            fir.append(heights[i] - heights[i-6])
        t = [i for i in range(len(fir))]
        # plt.plot(t, fir)
        sig = [fir[i]**2 for i in range(len(fir))]
        t=[i for i in range(len(sig))]
        # plt.plot(t, sig)
        window = int(0.15 * fs)
        denoised = []
        for i in range(window):
            denoised.append(0)
        for i in range(window, len(sig)):
            new = 0
            for m in range(window):
                new += sig[i - m]
            denoised.append(new / window)
        print(record)
        t = [i for i in range(len(denoised))]
        # plt.plot(t, denoised)
        normalized = []
        ymin= min(denoised)
        ymax=max(denoised)
        for i in range(len(denoised)):
            normalized.append((denoised[i]-ymin)/(ymax-ymin))
        t = [i for i in range(len(normalized))]
        plt.plot(t, normalized)

        potentials = []
        i=0
        while i<len(normalized):
            # print(i)
            if normalized[i] > 0.1:
                potentials.append(i)
                for m in range(i, len(normalized)):
                    if normalized[m] < 0.01:
                        i = m
                        break
            i += 1
        # print("done")
        locations=[]
        peaks=[]
        sdiffs=[]
        slope_heights=[]
        hi = []
        for m in potentials:
            # print(locations)
            locations, peaks, count, sdiffs, slope_heights = rpeakdetection.locate_r_peaks(heights, fs, round(0.35*fs), False, locations, peaks, slope_heights, sdiffs, m+100, m)
            hi. append(heights[m])

        plt. scatter(potentials, hi, color="red", marker='x')
        # plt.scatter(locations, peaks, color="red", marker="x")
        # plt.show()
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
        print(end - start)
        if excel:
            ws1.append((record, len(locations), len(ref_locations), TP, FP,
                                       FN, sensitivty, pp, DER, end-start))
    except FileNotFoundError:
        continue
if excel:
    wb.save(name)