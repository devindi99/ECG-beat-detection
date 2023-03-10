import numpy as np
from beatdetection import ptemplate
from beatdetection import filters
from beatdetection import beatpair
from beatdetection import rpeakdetection
from beatdetection import folderhandling


def test1():
    y = ptemplate.qrs(np.array([1, 2, 3, 4, 5, 6]), 2)
    assert (type(y) == np.int64)
