from beatdetection import rpeakdetection
from beatdetection import ptemplate
import matplotlib
from beatdetection import folderhandling
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtaidistance import dtw

folder = "Attempt2/"
folderhandling.mkdir_p(folder)

for record in range(100, 235):
    try:

        output_dir = folder + str(record)
        folderhandling.mkdir_p(output_dir)
        path = 'D:\Semester 6\Internship\mit-bih-arrhythmia-database-1.0.0/'
        locations, peaks = rpeakdetection.locate_r_peaks(record, path, 30 * 60)
        heights, fs = rpeakdetection.read_annotations(record, path)
        print(record)
        template = ptemplate.create_template([record], path, 1*60)
        t = [m for m in range(len(template))]
        plt.figure()
        plt.plot(t, template)
        filename = str(record) + "template.png"
        imagepath = output_dir + "/" + filename
        plt.savefig(imagepath)
        plt.close()
        pwaves = ptemplate.p_waves(record, path, 30*60)
        for m in range(len(pwaves)):
            p = pwaves[m]
            distance = dtw.distance(p, template)
            if distance > 2.5:
                plt.figure()
                T = [m for m in range(len(p))]
                plt.plot(T, p)
                filename = str(record) + "distance " + str(distance) + ".png"
                imagepath = output_dir + "/" + filename
                plt.savefig(imagepath)
                plt.close()
    except FileNotFoundError:
        continue

