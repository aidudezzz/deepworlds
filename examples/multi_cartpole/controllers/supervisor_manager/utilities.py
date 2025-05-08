import numpy as np
import matplotlib.pyplot as plt


def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
    """
    Normalizes value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :type value: float
    :param minVal: value's min value, value ∈ [minVal, maxVal]
    :type minVal: float
    :param maxVal: value's max value, value ∈ [minVal, maxVal]
    :type maxVal: float
    :param newMin: normalized range min value
    :type newMin: float
    :param newMax: normalized range max value
    :type newMax: float
    :param clip: whether to clip normalized value to new range or not, defaults to False
    :type clip: bool, optional
    :return: normalized value ∈ [newMin, newMax]
    :rtype: float
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
    Uses matplotlib to plot data.

    :param data: List of data to plot
    :type data: list
    :param xLabel: Label on x axis
    :type xLabel: str
    :param yLabel: Label on y axis
    :type yLabel: str
    :param plotTitle: Plot title
    :type plotTitle: str
    :param save: Whether to save plot automatically or not, defaults to False
    :type save: bool, optional
    :param saveName: Filename of saved plot, defaults to None
    :type saveName: str, optional
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
