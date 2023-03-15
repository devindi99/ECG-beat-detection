from beatdetection import rpeakdetection
from beatdetection import ptemplate
import matplotlib
from beatdetection import folderhandling
from beatdetection import beatpair
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtaidistance import dtw
from openpyxl import Workbook

folder = "Attempt3/"
folderhandling.mkdir_p(folder)
check_accuracy = int(input("Accuracy checker ON/ OFF? "))
excel = int(input("Write accuracy values to a XL sheet? "))
if excel:
    wb = Workbook()
    name = input("Enter workbook name: ")
    sheet = input("Enter sheet name: ")
    ws1 = wb.create_sheet("Mysheet")
    ws1.title = sheet
    # wb.save("dev.xlsx")


for record in range(100, 235):
    try:

        output_dir = folder + str(record)
        folderhandling.mkdir_p(output_dir)
        path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
        locations, peaks, time, count = rpeakdetection.locate_r_peaks(record, path, 30 * 60, True)
        heights, fs = rpeakdetection.read_annotations(record, path)
        t = [i for i in range(len(heights))]
        plt.plot(t, heights)

        if check_accuracy:
            ref_locations, ref_annotations, a_fib = beatpair.ref_annotate(record, path, count)
            TP, FP, FN, sensitivty, pp, DER = beatpair.accuracy_check(ref_locations, ref_annotations, locations, peaks,
                                                                      True, True)
            if excel:
                ws1.append((record, len(locations), len(ref_locations), TP, FP,
                                       FN, sensitivty, pp, DER, time))
        print(record)
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

