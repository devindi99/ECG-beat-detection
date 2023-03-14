import numpy as np
from beatdetection import ptemplate
from beatdetection import filters
from beatdetection import beatpair
from beatdetection import rpeakdetection
from beatdetection import folderhandling


def test_p_template():
    y1 = ptemplate.qrs(np.array([1, 2, 3, 4, 5, 6]), 2)
    assert (type(y1) == np.int64)

    y2 = ptemplate.extract_p_waves(110, 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/', 30)
    assert (y2 == None)

    y3 = ptemplate.create_template([100, 110, 109], 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/', 20)
    assert (type(y3) == np.ndarray)

    y4 = ptemplate.p_waves(100, 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/', 20)
    assert (type(y4) == list)


def test_filters():
    y1 = filters.Low_pass([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    assert (type(y1) == np.ndarray)

    y2 = filters.delay_com(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), 5)
    assert (type(y2) == np.ndarray)

    y3 = filters.iir(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), 2.3)
    assert (type(y3) == np.ndarray)

    y4 = filters.sqr_fil(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))
    assert (type(y4) == np.ndarray)

    y5 = filters.l_deriv(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))
    assert (type(y5) == np.ndarray)

    y6 = filters.mov_win(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))
    assert (type(y6) == np.ndarray)

    y7 = filters.high_pass(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))
    assert (type(y7) == np.ndarray)


def test_beat_pair():
    y1 = beatpair.beat_pair(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), 450)
    assert (type(y1) == np.ndarray)

    y2 = beatpair.read_annotations(100, 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/')
    assert (type(y2) == tuple)

    y3 = beatpair.test_annotate(100, 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/', 10)
    assert (type(y3) == tuple)


def test_r_peak_detection():
    y1 = rpeakdetection.read_annotations(100, 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/')
    assert (type(y1) == tuple)

    y2 = rpeakdetection.locate_r_peaks(100, 'D:\\Semester 6\\Internship\\mit-bih-arrhythmia-database-1.0.0/', 10)
    assert (type(y2) == tuple)

    y3 = rpeakdetection.initial(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 360)
    assert (type(y3) == bool)

    y4 = rpeakdetection.max_min_slopes(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 1002)
    assert (type(y4) == tuple)

    y5 = rpeakdetection.third_criterion(2.356, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    assert (type(y5) == np.bool_)

    y6 = rpeakdetection.teeta_diff([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 150)
    assert (type(y6) == float)

    y7 = rpeakdetection.first_criterion(2.356, 2.5544)
    assert (type(y7) == bool)

    y8 = rpeakdetection.second_criterion(3.25615, True, 400)
    assert (type(y8) == bool)

    y9 = rpeakdetection.max_slope_difference(1.235, 8.548, 1.234, 5.264, 6.346, 4.235)
    assert (type(y9) == tuple)

    y10 = rpeakdetection.s_min(5.234, 1.235, 6.235, 0.2358)
    assert (type(y10) == tuple)


def test_folder_handling():
    y1 = folderhandling.mkdir_p('D:\\Semester 6\\Internship\\test')
    assert (y1 is None)
