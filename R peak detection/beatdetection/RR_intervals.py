from beatdetection import rpeakdetection

record = int(input("Enter record number: "))

path = 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/'
heights, fs = rpeakdetection.read_annotations(record, path)
locations, peaks, time, count = rpeakdetection.locate_r_peaks(heights, fs, 30 * 60, True)
RR = []

for m in range(len(locations)-1):
    RR.append(locations[m+1]-locations[m])
    print(locations[m+1]-locations[m])
