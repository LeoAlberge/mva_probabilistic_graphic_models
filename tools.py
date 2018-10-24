import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, labels=None):
    """
    Plot 2D data
    """

    cols, marks = ["red", "blue"], [".", "+"]

    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], marker="x")
        return

    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels == l, 0], data[labels == l, 1], c=cols[i], marker=marks[i])


def plot_boundary(data, f, step=20, alpha_c=1):
    """
    Plot the boundary associated to a binary valued function f
    """

    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 256, alpha=alpha_c)


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """
    Create a grid in the form of a list of points stored in a 2D array
    """

    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:, 0]),  np.min(data[:, 0]), np.max(data[:, 1]), np.min(data[:, 1])

    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)*1./step), np.arange(ymin, ymax, (ymax-ymin)*1./step))
    grid = np.c_[x.ravel(), y.ravel()]

    return grid, x, y
