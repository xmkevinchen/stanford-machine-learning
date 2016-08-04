import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats


def estimate_gaussian(X):
    '''
    Estimate the paramters of a Gaussian distribution using the data in X.
    '''

    mu = np.mean(X, axis=0)
    var = np.var(X, ddof=1, axis=0)

    return mu, var


def multivariate_gaussian(X, mean, variance):

    # convinience way from scipy.stats module
    # return scipy.stats.multivariate_normal.pdf(X, mu, variance)

    k = len(mean)

    if variance.ndim == 1:
        variance = np.diag(variance)

    X_mu = X - mean
    norm = X_mu.dot(np.linalg.pinv(variance)) * X_mu
    b = np.exp(-0.5 * np.sum(norm, axis=1))
    a = ((2 * np.pi) ** k * np.linalg.det(variance)) ** -0.5

    p = a * b
    return p

def visualize_fit(X, mean, variance):
    x = y = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(x, y)
    Z = multivariate_gaussian(np.c_[X1.reshape(-1,1), X2.reshape(-1,1)], mean, variance)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.hold(True)

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, 10 ** (np.arange(-20, 0, 3, dtype=float).T))

    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

    plt.hold(False)

def select_threshold(yval, pval):
    '''
    Find the best threshold (epsilon) to use for selecting outliers
    '''

    bestF1 = 0
    bestEpsilon = 0
    stepsize = (np.max(pval) - np.min(pval)) / 1000.0
    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        predictions = (pval < epsilon).astype(int)
        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))

        if tp + fp == 0 or tp + fn == 0:
            continue

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = 2 * prec * rec / (prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

        return bestEpsilon, bestF1


if __name__ == '__main__':
    data = sio.loadmat('ex8data1.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    mean, variance = estimate_gaussian(X)
    p = multivariate_gaussian(X, mean, variance)

    visualize_fit(X, mean, variance)

    pval = multivariate_gaussian(Xval, mean, variance)
    epsilon, F1 = select_threshold(yval, pval)

    outliers = np.where(p < epsilon)

    plt.hold(True)
    plt.plot(X[outliers, 0], X[outliers, 1], 'o', linewidth=2, markersize=10, markerfacecolor='None', markeredgecolor='r')
    plt.hold(False)
    plt.show()

    data = sio.loadmat('ex8data2.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    mean, variance = estimate_gaussian(X)

    # Training set
    p = multivariate_gaussian(X, mean, variance)

    # Cross-validation set
    pval = multivariate_gaussian(Xval, mean, variance)

    # Find the best threshold
    epsilon, F1 = select_threshold(yval, pval)

    print('epsilon = {}'.format(epsilon))
