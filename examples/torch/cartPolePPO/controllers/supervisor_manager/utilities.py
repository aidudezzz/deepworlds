import numpy as np
import matplotlib.pyplot as plt


def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
    """
    Normalize value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :param minVal: value's min value, value ∈ [minVal, maxVal]
    :param maxVal: value's max value, value ∈ [minVal, maxVal]
    :param newMin: normalized range min value
    :param newMax: normalized range max value
    :param clip: whether to clip normalized value to new range or not
    :return: normalized value ∈ [newMin, newMax]
    """
    value = float(value)
    minVal = float(minVal)
    maxVal = float(maxVal)
    newMin = float(newMin)
    newMax = float(newMax)

    if clip:
        return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
    else:
        return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax


def plotData(data, xLabel, yLabel, plotTitle, save=False, saveName=None):
    """
    Use matplotlib to plot data.

    :param data: list of data
    :param xLabel: str, label on x axis
    :param yLabel: str, label on y axis
    :param plotTitle: str, plot title
    :param save: bool, whether to save plot automatically or not
    :param saveName: str, filename of saved plot
    :return: None
    """
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel=xLabel, ylabel=yLabel,
           title=plotTitle)
    ax.grid()
    if save:
        if saveName is not None:
            fig.savefig(saveName)
        else:
            fig.savefig("figure")
    plt.show()