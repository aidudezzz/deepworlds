import numpy as np
import matplotlib.pyplot as plt


def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalizes value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :type value: float
    :param min_val: value's min value, value ∈ [min_val, max_val]
    :type min_val: float
    :param max_val: value's max value, value ∈ [min_val, max_val]
    :type max_val: float
    :param new_min: normalized range min value
    :type new_min: float
    :param new_max: normalized range max value
    :type new_max: float
    :param clip: whether to clip normalized value to new range or not, defaults to False
    :type clip: bool, optional
    :return: normalized value ∈ [new_min, new_max]
    :rtype: float
    """
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max


def plot_data(data, x_label, y_label, plot_title, save=False, save_name=None):
    """
    Uses matplotlib to plot data.

    :param data: List of data to plot
    :type data: list
    :param x_label: Label on x axis
    :type x_label: str
    :param y_label: Label on y axis
    :type y_label: str
    :param plot_title: Plot title
    :type plot_title: str
    :param save: Whether to save plot automatically or not, defaults to False
    :type save: bool, optional
    :param save_name: Filename of saved plot, defaults to None
    :type save_name: str, optional
    """
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel=x_label, ylabel=y_label,
           title=plot_title)
    ax.grid()
    if save:
        if save_name is not None:
            fig.savefig(save_name)
        else:
            fig.savefig("figure")
    plt.show()
