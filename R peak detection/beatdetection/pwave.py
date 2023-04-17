from beatdetection import rpeakdetection
from beatdetection import ptemplate
import matplotlib
from beatdetection import folderhandling
from beatdetection import beatpair
from beatdetection import filters
from beatdetection import recorrect
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtaidistance import dtw
from openpyxl import Workbook
import time
import pdb
import pywt
import numpy as np

def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


level = 4



folder = "Attempt3/"
folderhandling.mkdir_p(folder)
# check_accuracy = int(input("Accuracy checker ON/ OFF? "))
# excel = int(input("Write accuracy values to a XL sheet? "))
# if excel:
#     wb = Workbook()
#     name = input("Enter workbook name: ")
#     sheet = input("Enter sheet name: ")
#     ws1 = wb.create_sheet("Mysheet")
#     ws1.title = sheet

d = round(0.4*360)
for record in range(108, 235):

    try:
        remove = [102, 104, 107, 217]
        if record in remove:
            continue
        output_dir = folder + str(record)
        folderhandling.mkdir_p(output_dir)
        path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
        heights, fs = rpeakdetection.read_annotations(record, path)

        heights = filters.Low_pass(heights)
        heights = filters.iir(heights, 2000)

        # heights = filters.Low_pass(heights)
        # heights = filters.iir(heights, 2000)
        t = [i for i in range(len(heights))]
        plt.plot(t, heights)
        # coeffs = pywt.wavedec(heights, 'db1', level=level)
        # A4 = wrcoef(heights, 'a', coeffs, 'db1', level)
        # D4 = wrcoef(heights, 'd', coeffs, 'db1', level)
        # D3 = wrcoef(heights, 'd', coeffs, 'db1', 3)
        # D2 = wrcoef(heights, 'd', coeffs, 'db1', 2)
        # D1 = wrcoef(heights, 'd', coeffs, 'db1', 1)
        #
        # heights = A4+ D4
        # print(heights)
        # t = [i for i in range(len(heights))]
        # plt.plot(t, heights)
        print(record)
        l = []
        p = []
        start = time.time()

        ref_locations, ref_annotations, a_fib = beatpair.ref_annotate(record, path, fs)

        plt.scatter(ref_locations, ref_annotations, color="black", marker="x")
        # plt.show()

        template = ptemplate.create_template([record], ref_locations, heights)
        t = [m for m in range(len(template))]


        filename = str(record) + "template.png"
        imagepath = output_dir + "/" + filename
        # plt.savefig(imagepath)
        # plt.close()
        pwaves, ploc, ori = ptemplate.p_waves(record, ref_locations, heights)

        for m in range(len(pwaves)):
            p = pwaves[m]
            # state = ptemplate.check_window(d, ori[m], template, ploc[m])
            # if state:
            #     # print(state)
            #     plt.plot(ploc[m], ori[m], color="red")
            #
            #     pass
            distance = dtw.distance(p, template)
            if distance > 1.25:
                # plt.figure()
                # T = [m for m in range(len(p))]
                # plt.plot(T, p)
                # plt.title(distance)
                # plt.show()
                plt.plot(ploc[m], ori[m], color="red")
                filename = str(record) + "distance " + str(distance) + ".png"
                imagepath = output_dir + "/" + filename
                # plt.savefig(imagepath)
                # plt.close()
            else:
                # plt.figure()
                plt.plot(ploc[m], ori[m], color="green")
                # plt.title(distance)
                # plt.show()

        plt.figure()
        plt.plot(t, template)
        plt.show()
    except FileNotFoundError:
        continue
# if excel:
#     wb.save(name)

