from matplotlib import pyplot as plt
import numpy as np


def quickplot(lst, xaxis, yaxis):
    plt.plot(lst)
    plt.xlabel(f"{xaxis}")
    plt.ylabel(f"{yaxis}")
    plt.show()


def rollingplot(lst, xaxis, yaxis, roll_length):
    values = np.convolve(np.array(lst).flatten(), np.ones(roll_length), mode="valid")
    plt.plot(values)
    plt.xlabel(f"{xaxis}")
    plt.ylabel(f"{yaxis}")
    plt.show()


def moving_average(data, len_window):
    data_sum = np.cumsum(data)
    return (data_sum[len_window:] - data_sum[:-len_window]) / len_window


def rollingplot(lst, xaxis, yaxis, roll_length):
    """
    Plots the moving average of lst with window size roll_length

    Params:
    lst: the data, array-like
    xaxis: label for the x axis, string
    yaxis: label for the y axis, string
    roll_length: moving average window
    """
    values = moving_average(lst, roll_length)
    plt.plot(values)
    plt.xlabel(f"{xaxis}")
    plt.ylabel(f"{yaxis}")
    plt.show()
