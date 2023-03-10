# import the WFDB package
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from BaselineRemoval import BaselineRemoval
import time

record=input("Enter record number: ")
start=time.time()
def readAnnotations(name):
    # path = 'D:\Semester 6\Internship\mit-bih-noise-stress-test-database-1.0.0/' + str(name)
    # path = "D:\Semester 6\Internship\mit-bih-arrhythmia-database-p-wave-annotations-1.0.0/" + str(name)
    path ="D:\Semester 6\Internship\mit-bih-arrhythmia-database-1.0.0/"+ str(name)
    # path ="D:\Semester 6\Internship\mit-bih-atrial-fibrillation-database-1.0.0/"+ str(name)
    signals, fields = wfdb.rdsamp(path, channels=[0])
    # annotations = wfdb.rdann(path, 'pwave')
    annotations = wfdb.rdann(path, 'atr')
    heights = [signals[i][0] for i in range(len(signals))]
    # resampled = signal.resample_poly(heights, 250, 360)

    time_re = [i for i in range(0, len(heights))]

    plt.figure()
    # plt.plot(time_re, heights)

    # plt.show()
    baseObj = BaselineRemoval(heights)
    Zhangfit_output = baseObj.ZhangFit()
    plt.plot(time_re, heights)
    return annotations, signals,heights, fields["fs"]



#functions for checking the first criterion


def max_min_slopes(n, samples, fs):   #n= number of current sample, samples= sample values of ECG signal
    a = round(0.027*fs)
    b = round(0.063*fs)
    Lslopes = []
    Rslopes = []
    Lheights = []
    Rheights = []
    for k in range(a, b+1):
        if n+k >= len(samples) or n-k < 0:
            break
        Lslopes.append((samples[n]-samples[n-k])/k)
        Rslopes.append((samples[n]-samples[n+k])/(-1*k))
        Lheights.append(np.absolute(samples[n] - samples[n - k]))
        Rheights.append(np.absolute(samples[n] - samples[n + k]))
    return max(Rslopes), min(Rslopes), max(Lslopes), min(Lslopes),max(Rheights), max(Lheights)




def max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight):
    # print(max_R, "  ", min_L)
    if max_L - min_R > max_R - min_L: #positive peak
        return max_L - min_R , max_Lheight
    else: #negatuve peaks
        return max_R - min_L, max_Rheight

def teeta_diff(peaks,fs):
    s_avg = np.average(np.absolute(peaks[-8:]))

    if s_avg > 20.480 / fs:
        return 7.680 / fs
    elif 2.800 / fs < s_avg < 20.480 / fs:
        return 4.352 / fs
    else:
        return 3.840 / fs

def average_RR(RR_Intervals):
    return np.average(RR_Intervals[-8:])

def first_criterion(teeta_diff, Sdiff_max):

    return Sdiff_max > teeta_diff



#functions for checking the second criterion


def S_min(max_R, min_R, max_L, min_L):
    if max_L - min_R > max_R - min_L:
        Smin = min (np.abs(max_L), np.abs(min_R))
        state = (np.sign(max_L) == -1 * np.sign(min_R))

    else:
        Smin = min (np.abs(max_R), np.abs(min_L))
        state = (np.sign(max_R) == -1 * np.sign(min_L))
    return Smin, state


def second_criterion(Smin, state, fs):

    if Smin > 1.536/fs and state:
        return True
    else:
        return False


#function for checking third criterion


def third_criterion(cur_height, peaks):
    try:
        h_avg = np.average(np.absolute(peaks[-8:]))
    except IndexError:
        h_avg = np.average(np.absolute(peaks[:]))

    return np.abs(cur_height) > h_avg * 0.4

def beatpair(refannot, testannot, fs=360):
    refannot = refannot.flatten().reshape((len(refannot), -1))
    testannot = testannot.flatten().reshape((len(testannot),-1))
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
                T = refannstruct[TIndex, 0]
                t = tdash

            elif abs(tdash - T) < abs(T - t) and abs(tdash - T) <= threshold:
                pairelement = [-1, tindex - 1]  # 0 = O, 1 = X, 2 = paired
                pairedlist.append(pairelement)
                pairelement = [TIndex, tindex]
                pairedlist.append(pairelement)
                TIndex += 1
                tindex += 1
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
                t = testannstruct[tindex, 0]
                T = Tdash

            elif abs(Tdash - t) < abs(t - T) and abs(Tdash - t) <= threshold:
                pairelement = [TIndex - 1, -1]  # % 0 = O, 1 = X, 2 = paired
                pairedlist.append(pairelement)
                pairelement = [TIndex, tindex]
                pairedlist.append(pairelement)
                TIndex += 1
                tindex += 1
                T = refannstruct[TIndex, 0]
                t = testannstruct[tindex, 0]

            elif (abs(Tdash - t) > threshold) and (abs(t - T) > threshold):
                pairelement = [TIndex - 1, -1]
                pairedlist.append(pairelement)
                T = Tdash

    return np.asarray(pairedlist)



def Low_pass(x):
    #x=np.ndarray.flatten(x)
    y=np.copy(x)
    y[0]=x[0]
    y[1]=2*y[0]+x[1]
    for i in range(2,len(x)):
        if 2<=i<6:
            y[i]=2*y[i-1]+x[i]-y[i-2]
        if 6<=i<12:
            y[i]=2*y[i-1]+x[i]-y[i-2]-2*x[i-6]
        if i>=12:
            y[i]=2*y[i-1]+x[i]-y[i-2]-2*x[i-6]+x[i-12]
    return delay_com(y/36,5)

def delay_com(y, d):
    a = y[d:]
    b = np.zeros(d)
    return np.append(y[d:], np.zeros(d))


def IIR(x, Fc=0.65):
    delta = 0.6
    Fs = 360
    g = np.tan(np.pi * Fc * 0.001 / Fs)
    y = np.copy(x)

    g1 = 1 + 2 * delta * g + g ** 2
    g2 = 2 * g ** 2 - 2
    g3 = 1 - 2 * delta * g + g ** 2

    b = g ** 2 / g1
    a2 = g2 / g1
    a3 = g3 / g1

    y[1] = b * x[1] + (3 * b - a3) * x[0] - a2 * y[0]

    for i in range(2, len(x)):
        y[i] = b * (x[i] + x[i - 1] + x[i - 2]) - a2 * y[i - 1] - a3 * y[i - 2]

    y = y[::-1]
    y[0] = x[0]
    y[1] = b * x[1] + (3 * b - a3) * x[0] - a2 * y[0]

    for i in range(2, len(x)):
        y[i] = b * (x[i] + x[i - 1] + x[i - 2]) - a2 * y[i - 1] - a3 * y[i - 2]
    y = delay_com(y, 30)
    y = np.subtract(x, y)

    return y

def High_pass(x):
    #x=np.ndarray.flatten(x)
    y=np.copy(x)
    for i in range(len(x)):
        if i<32:
            y[i]=0
        if i>=32:
            y[i]= (-y[i-1]-x[i]+32*x[i-16]-x[i-17]+x[i-32])/32
    return delay_com(y,16)

def l_deriv(x):
    y=np.copy(x)
    n=8
    y[0]=n*x[0]-x[2]
    y[1]=n*x[1]-x[3]-n*x[0]
    for i in range(2,len(x)-2):
        y[i]=x[i-2]-n*x[i-1]+n*x[i]-x[i+2]
    for i in range(len(x)-2,len(x)):
        y[i]=x[i-2]-n*x[i-1]+n*x[i]
    return delay_com(y/12,0)

def sqr_fil(x):
    return np.square(x)

def mov_win(x):
    N=60
    y=np.zeros(len(x))
    for i in range(N-1,len(x)):
        y[i]=(sum(x[i-N-2:i+1]))/N
    return y


def QRS(r, n):
    global Onset_offsets
    global Borders
    W = 30
    n = int(n)
    Wn1 = r[int(max(n - W, 0)):n]
    Wn2 = r[n:int(min(n + W, len(r)))]
    """searching for the first half"""
    Lmin = min(Wn1)
    Lmax = max(Wn2)
    LA = abs(Lmax - Lmin)
    Lmin += (LA / 100)
    Lmax -= (LA / 100)
    Qb = n - (len(Wn1) - max(np.where(Wn1 <= Lmin)[-1])) - 5
    return Qb




# annotations, signals, heights, fs = readAnnotations(name)
annotations, signals, heights, fs = readAnnotations(record)
Refannotations=[]
Reflocations=[]
remove_sym=["+","|","~","x","]","[","U", " MISSB", "PSE", "TS", "T", "P","M","!"]
AFIB = {}
afib_keys=[]
afib_beats=[]
for i in range(len(annotations.sample)):
    s = 0
    e = 0
    if annotations.symbol[i] == "[":
        s = annotations.sample[i]
        while (annotations.symbol[i] != "]"):
            e = annotations.sample[i]
            i+=1
        AFIB[str(s)]=str(e)
        # afib_beats.append(m for m in range(s,e))
        for m in range(s,e+1):
            afib_beats.append(m)

    if annotations.symbol[i] in remove_sym:
        continue
    else:
        Refannotations.append(heights[annotations.sample[i]])
        Reflocations.append(annotations.sample[i])
peaks = [Refannotations[0]]
locations = [Reflocations[0]]
wrong=[]
wrong_loc=[]
slope_heights = []
Sdiffs = []
RR_Intervals=[]
a = round(0.027 * fs)
b = round(0.063 * fs)
c = round(0.3125* fs)

for i in range(1,8):
    peaks.append(Refannotations[i])
    locations.append(Reflocations[i])
    RR_Intervals.append(locations[-1]-locations[-2])
# print(RR_Intervals)
# print(locations)
# c = round(average_RR(RR_Intervals))
for i in range(b, 3 * fs):
    max_R, min_R, max_L, min_L, max_Rheight, max_Lheight = max_min_slopes(i, heights, fs)
    Sdiff_max, max_height = max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight)
    slope_heights.append(max_height)
    Sdiffs.append(Sdiff_max)
TP=0
FP=0
FN=0
TP_list=[]
FP_list=[]
FN_list=[]
TP_list_loc=[]
FP_list_loc=[]
FN_list_loc=[]
callibration_time= 1*60*fs
for i in range(b, len(heights) + 1 - b):
    max_R, min_R, max_L, min_L, max_Rheight, max_Lheight = max_min_slopes(i, heights, fs)
    Sdiff_max, max_height = max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight)
    teeta = teeta_diff(Sdiffs, fs)
    Smin, state = S_min(max_R, min_R, max_L, min_L)
    QRS_complex = first_criterion(teeta, Sdiff_max)and second_criterion(Smin, state, fs)and third_criterion(max_height, slope_heights)
    # if QRS_complex:
        # element = max(np.absolute(heights[i - a:i + a + 1]))
        # loc = np.where(np.absolute(heights[i - a:i + a + 1]) == element)
        # loc = loc[0][0] + i - a
        # locations.append(loc)
        # peaks.append(heights[loc])
        # slope_heights.append(max_height)
        # Sdiffs.append(Sdiff_max)
        # if loc - c > locations[-1]:
        #     locations.append(loc)
        #     peaks.append(heights[loc])
        #     slope_heights.append(max_height)
        #     Sdiffs.append(Sdiff_max)
        # else:
        #     if Sdiff_max > Sdiffs[-1]:
        #         peaks[-1] = heights[loc]
        #         locations[-1] = loc
        #         slope_heights[-1] = max_height
        #         Sdiffs[-1] = Sdiff_max

    if QRS_complex:
        # peaks.append(heights[i])
        # locations.append(i)
        if i - c > locations[-1]:
            locations.append(i)
            peaks.append(heights[i])
            slope_heights.append(max_height)
            Sdiffs.append(Sdiff_max)

        else:
            if Sdiff_max > Sdiffs[-1]:
                peaks[-1] = heights[i]
                locations[-1] = i
                slope_heights[-1] = max_height
                Sdiffs[-1] = Sdiff_max

# RR=[]
# print(c)
# n = len(locations)
# for m in range(n - 40, n - 10):
#     RR.append(locations[m + 1] - locations[m])
# c = np.average(RR) - 150
# print(c)



# for m in range(len(AFIB)):
#     afib_keys.append(AFIB.keys()[m])
# afib_keys=AFIB.keys()
# print(len(heights) + 1 - b)
# while i < len(heights) + 1 - b:
#     # print("   ",i)
#     # print(c)
#     # c=np.average(RR_Intervals)-100
#     # if i % callibration_time == 0:
#     #     c = 0
#     #     RR = []
#     #     n = len(locations)
#     #     for m in range(n-60,n -40):
#     #         RR.append(locations[m + 1] - locations[m])
#     #     #     c+=(locations[m+1]-locations[m])/20
#     #     # c-=100
#     #
#     #     c = np.average(RR) - 100
#     # if i % callibration_time == 0:
#     #     c = 0
#     #     RR = []
#     #     n = len(locations)
#     #
#     #     for m in range(n - 40, n - 10):
#     #         RR.append(locations[m + 1] - locations[m])
#     #     #     c+=(locations[m+1]-locations[m])/20
#     #     # c-=100
#     #     c = np.average(RR) - 110
#     #     print(RR)
#
#     # if i in afib_beats:
#     #     print("scsdcdscd", i)
#     #     key=afib_keys[afib_count]
#     #     i=AFIB[key]+1
#     #     print("             ", i)
#     #     afib_count+=1
#     #     continue
#     if str(i) in AFIB.keys():
#         print("scsdcdscd",i)
#         i=int(AFIB[str(i)])+1
#         print("             ",i)
#         continue
#
#     max_R, min_R, max_L, min_L, max_Rheight, max_Lheight = max_min_slopes(i, heights, fs)
#     Sdiff_max, max_height = max_slope_difference(max_R, min_R, max_L, min_L, max_Lheight, max_Rheight)
#     teeta = teeta_diff(Sdiffs, fs)
#     Smin, state = S_min(max_R, min_R, max_L, min_L)
#     QRS_complex = first_criterion(teeta, Sdiff_max)and second_criterion(Smin, state, fs)and third_criterion(max_height, slope_heights)
#     if QRS_complex:
#         # print("wtrses")
#         element = max(np.absolute(heights[i - a:i + a + 1]))
#         loc = np.where(np.absolute(heights[i - a:i + a + 1]) == element)
#         loc = loc[0][0] + i -a
#         # locations.append(loc)
#         # print(loc)
#         # peaks.append(heights[loc])
#         # slope_heights.append(max_height)
#         # Sdiffs.append(Sdiff_max)
#         # i = locations[-1]
#         # print(i)
#         if loc - c > locations[-1]:
#             locations.append(loc)
#             # print(loc)
#             peaks.append(heights[loc])
#             slope_heights.append(max_height)
#             Sdiffs.append(Sdiff_max)
#             i = locations[-1]
#
#         else:
#             if Sdiff_max > Sdiffs[-1]:
#                 peaks[-1] = heights[loc]
#                 locations[-1] = loc
#                 # print("   xas",loc)
#                 slope_heights[-1] = max_height
#                 Sdiffs[-1] = Sdiff_max
#                 i=locations[-1]
#             else:
#                 pass
#     i+=1+a
#
#
#     # if QRS_complex:
#     #     # peaks.append(heights[i])
#     #     # locations.append(i)
#     #     if i - c > locations[-1]:
#     #         locations.append(i)
#     #         peaks.append(heights[i])
#     #         slope_heights.append(max_height)
#     #         Sdiffs.append(Sdiff_max)
#     #         RR_Intervals.append(i - locations[-2])
#     #         c = round(average_RR(RR_Intervals))
#     #     else:
#     #         if Sdiff_max > Sdiffs[-1]:
#     #             peaks[-1] = heights[i]
#     #             locations[-1] = i
#     #             slope_heights[-1] = max_height
#     #             Sdiffs[-1] = Sdiff_max
#     #             RR_Intervals[-1] = i - locations[-2]
#     #             c = round(average_RR(RR_Intervals))
end=time.time()
print(end-start)

# print(RR_Intervals)
# print(locations)
onsets = []
onset_heights = []
locations=np.array(locations[7:])
peaks=np.array(peaks[7:])
y1=Low_pass(heights)
y = IIR(y1, 100)
Base = np.mean(y)
ybp=High_pass(y)
ysq=sqr_fil(l_deriv(ybp-Base))
ymv=mov_win(ysq)
ydr=l_deriv(ymv)
for m in range(len(locations) - 1):
    onsets.append(QRS(ydr, locations[m]))
    onset_heights.append(heights[QRS(ydr, locations[m])])

Reflocations=np.array(Reflocations)
Refannotations=np.array(Refannotations)
pairedlist = beatpair(Reflocations, locations)

for i in range(len(pairedlist)):
    # print(pairedlist[i])
    if pairedlist[i][0] == -1:
        FP += 1
        FP_list.append(peaks[pairedlist[i][1]])
        FP_list_loc.append(locations[pairedlist[i][1]])
    elif pairedlist[i][1] == -1:
        FN += 1
        FN_list.append(Refannotations[pairedlist[i][0]])
        FN_list_loc.append(Reflocations[pairedlist[i][0]])


    else:
        TP += 1
        TP_list.append(peaks[pairedlist[i][1]])
        TP_list_loc.append(locations[pairedlist[i][1]])
        # except IndexError:
        #     print(i)

sensitivty = TP * 100 / (TP + FN)
pp = TP * 100 / (TP + FP)
DER = (FP + FN) / len(Reflocations)
plt.scatter(TP_list_loc,TP_list,color="red",marker="x")
plt.scatter(FN_list_loc,FN_list,color="green",marker="x")
plt.scatter(FP_list_loc,FP_list,color="black",marker="x")
plt.scatter(onsets,onset_heights,color="purple",marker="x")
# plt.scatter(locations,peaks,color="red",marker="x")
print("TP: ",TP)
print("FP: ",FP)
print("FN: ",FN)
# print(locations)

# print(locations)
# Accurate_ann_loc = []
# Accurate_ann_height = []
# Incorrect_ann_loc = []
# Incorrect_ann_height = []
# MIT_ann = []
# MIT_loc = []
#
# correct = 0
# for i in range(len(locations)):
#     if locations[i] in Reflocations:
#         Accurate_ann_loc.append(locations[i])
#         Accurate_ann_height.append(peaks[i])
#         correct += 1
#     else:
#         Incorrect_ann_loc.append(locations[i])
#         Incorrect_ann_height.append(peaks[i])
#
# plt.scatter(locations, peaks, color="red", marker='x')

# plt.scatter(Accurate_ann_loc, Accurate_ann_height, color="red", marker='x')
# plt.scatter(Incorrect_ann_loc, Incorrect_ann_height, color="black", marker='o')
# plt.scatter(Reflocations,Refannotations, color="green", marker='x')
# for i in range(len(Reflocations)):
#     if Reflocations[i] in Accurate_ann_loc:
#         continue
#     else:
#         MIT_ann.append(heights[Reflocations[i]])
#         MIT_loc.append(Reflocations[i])

# plt.scatter(MIT_loc, MIT_ann, color="green", marker='x')






# pairedlist=beatpair(np.array(Reflocations), np.array(locations[7:]))
# for i in range(len(pairedlist)):
#     print(pairedlist[i])
plt.show()