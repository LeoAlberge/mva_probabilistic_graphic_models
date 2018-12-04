import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal


sns.set()


def initialize_random_parameters(K):
    # transition matrix
    a = np.random.rand(K, K)
    for i in range(K):
        a[i] *= 1 / np.sum(a[i])
    
    # initial probability
    pi = np.random.rand(K)
    pi *= 1 / np.sum(pi)
    return a, pi


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """
    Create a grid in the form of a list of points stored in a 2D array
    """

    if data is not None:
        xmax, xmin,  = np.max(data[:, 0]), np.min(data[:, 0])
        ymax, ymin = np.max(data[:, 1]), np.min(data[:, 1])
        
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)*1./step), 
                       np.arange(ymin, ymax, (ymax-ymin)*1./step))
    
    grid = np.c_[x.ravel(), y.ravel()]

    return grid, x, y


def plot_gaussian_mixture(data_train, data_test, means, covs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle('data sets visualization (means and covariances were computed in the previous homework)')

    grid, x_grid, y_grid = make_grid(data=data_train.values, step=100)

    ax1.axis([np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)])
    ax1.set_title('training set')

    for j in range(means.shape[0]):
        ax1.scatter(x=data_train.values[:, 0], y=data_train.values[:, 1], c='blue', marker='.')
        ax1.scatter(x=means[:, 0], y=means[:, 1], c='black', marker='X')
        ax1.contour(x_grid, y_grid, multivariate_normal.pdf(grid, mean=means[j],
                                                            cov=covs[j]).reshape(x_grid.shape),
                    levels=None, alpha=0.7, linestyles='solid', cmap='Blues')

    ax2.axis([np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)])
    ax2.set_title('test set')

    for j in range(means.shape[0]):
        ax2.scatter(x=data_test.values[:, 0], y=data_test.values[:, 1], c='red', marker='.')
        ax2.scatter(x=means[:, 0], y=means[:, 1], c='black', marker='X')
        ax2.contour(x_grid, y_grid, multivariate_normal.pdf(grid, mean=means[j],
                                                            cov=covs[j]).reshape(x_grid.shape),
                    levels = None, alpha=0.6, linestyles='solid', cmap='Reds')
        

def plot_hidden_states(data_train, data_test, K):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    ax1.set_title('HMM training set')
    ax2.set_title('GMM training set')

    sns.scatterplot(x="x", y="y", data=data_train, hue='states_hmm', 
                    palette=sns.color_palette("muted", K), ax=ax1)

    sns.scatterplot(x="x", y="y", data=data_train, hue='states_gmm', 
                    palette=sns.color_palette("muted", K), ax=ax2)

    ax3.set_title('HMM test set')
    ax4.set_title('GMM test set')

    sns.scatterplot(x="x", y="y", data=data_test, hue='states_hmm', 
                    palette=sns.color_palette("muted", K), ax=ax3)

    sns.scatterplot(x="x", y="y", data=data_test, hue='states_gmm', 
                    palette=sns.color_palette("muted", K), ax=ax4)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper left')

    fig.tight_layout()
    
    
def evaluate_models(data_train, data_test, gaussian_hmm, gaussian_mixture):
    print('HMM log likelihood (training set) : {0:.2f}'.format(gaussian_hmm.log_likelihood(data_train.values)))
    print('HMM log likelihood (test set) : {0:.2f}'.format(gaussian_hmm.log_likelihood(data_test.values)))
    print()

    gaussian_mixture.predict(data_train.values)
    print('GMM log likelihood (training set) : {0:.2f}'.format(gaussian_mixture.jfunc()))
    gaussian_mixture.predict(data_test.values)
    print('GMM log likelihood (test set) : {0:.2f}'.format(gaussian_mixture.jfunc()))
