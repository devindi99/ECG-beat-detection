import numpy as np


def Low_pass(
        x: list) -> np.ndarray:
    """

    :param x: sample heights
    :return: delayed version of the low pass filtered signal
    """
    y = np.copy(x)
    y[0] = x[0]
    y[1] = 2 * y[0] + x[1]
    for i in range(2, len(x)):
        if 2 <= i < 6:
            y[i] = 2 * y[i - 1] + x[i] - y[i - 2]
        if 6 <= i < 12:
            y[i] = 2 * y[i - 1] + x[i] - y[i - 2] - 2 * x[i - 6]
        if i >= 12:
            y[i] = 2 * y[i - 1] + x[i] - y[i - 2] - 2 * x[i - 6] + x[i - 12]
    return delay_com(y / 36, 5)


def delay_com(
        y: np.ndarray,
        d: int) -> np.ndarray:
    """

    :param y: list
    :param d:
    :return: numpy array
    """
    a = y[d:]
    b = np.zeros(d)
    return np.append(y[d:], np.zeros(d))


def iir(
        x: np.ndarray,
        Fc: float = 0.65) -> np.ndarray:
    """

    :param x: samples
    :param Fc: cutoff frequency
    :return: IIR filtered samples
    """
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


def high_pass(
        x: np.ndarray) -> np.ndarray:
    """

    :param x: samples
    :return: delayed version of the high pass filtered sample
    """
    y = np.copy(x)
    for i in range(len(x)):
        if i < 32:
            y[i] = 0
        if i >= 32:
            y[i] = (-y[i - 1] - x[i] + 32 * x[i - 16] - x[i - 17] + x[i - 32]) / 32
    return delay_com(y, 16)


def l_deriv(
        x: np.ndarray) -> np.ndarray:
    """

    :param x: samples
    :return: delayed version of the filtered samples
    """
    y = np.copy(x)
    n = 8
    y[0] = n * x[0] - x[2]
    y[1] = n * x[1] - x[3] - n * x[0]
    for i in range(2, len(x) - 2):
        y[i] = x[i - 2] - n * x[i - 1] + n * x[i] - x[i + 2]
    for i in range(len(x) - 2, len(x)):
        y[i] = x[i - 2] - n * x[i - 1] + n * x[i]
    return delay_com(y / 12, 0)


def sqr_fil(
        x: np.ndarray) -> np.ndarray:
    """

    :param x: samples
    :return: square filtered samples
    """
    return np.square(x)


def mov_win(
        x: np.ndarray) -> np.ndarray:
    """

    :param x: samples
    :return: filtered samples
    """
    N = 60
    y = np.zeros(len(x))
    for i in range(N - 1, len(x)):
        y[i] = (sum(x[i - N - 2:i + 1])) / N
    return y