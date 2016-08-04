import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.cm as cm
import os
import scipy

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = X_norm.std(axis=0, ddof=1)
    X_norm = X_norm / sigma

    return X_norm, mu, sigma

def pca(X):
    m, n = X.shape
    U = np.zeros(n)
    S = np.zeros(n)

    covariance = X.T.dot(X) / m;
    U, S, V = np.linalg.svd(covariance)
    return U, np.diag(S)

def project_data(X, U, K):
    U_reduced = U[:, range(K)]
    return X.dot(U_reduced)

def recover_data(Z, U, K):
    X_recover = Z.dot(U[:, range(K)].T)
    return X_recover

def pause():
    input("")

def plot_data_points(X, idx, K):
    n = K + 1
    palette = cm.colors.hsv_to_rgb(np.c_[np.arange(n).reshape(-1, 1) / n, np.ones((n, 2))])
    colors = palette[idx, :]

    ax = plt.gca()
    points = ax.scatter(X[:, 0], X[:, 1], s=100, edgecolors=colors, facecolors='None')
    plt.draw()

    return points


class KMeans(object):

    def __init__(self, X, K = 3, normalized=False):

        if normalized:
            self.X = self.feature_normalize(X)
        else:
            self.X = X

        self.K = K
        self.points = None

    def find_closest_centroids(self, centroids):
        K = centroids.shape[0]
        idx = np.zeros(self.X.shape[0], np.int8)

        for i in range(self.X.shape[0]):
            distances = np.zeros((K, 1))

            for k in range(K):
                distances[k] = np.sum((self.X[i, :] - centroids[k, :]) ** 2)

            idx[i] = distances.argmin()

        return idx

    def compute_centroids(self, idx):

        row, column = self.X.shape
        centroids = np.zeros((self.K, column))

        for k in range(self.K):
            indexes = np.where(idx == k)
            centroids[k, :] = np.mean(self.X[indexes, :], axis=1)

        return centroids

    def init_centroids(self):

        row, column = self.X.shape
        randidx = np.random.permutation(row)

        return self.X[randidx[:self.K], :]


    def run(self, initial_centroids=None, max_iters=10, plot_progress=False):

        if plot_progress:
            plt.figure()
            plt.hold(True)

        row, column = self.X.shape

        if initial_centroids is None:
            centroids = self.init_centroids()
        else:
            centroids = initial_centroids

        K, _ = centroids.shape
        previous_centroids = centroids


        for i in range(max_iters):
            print("K-Means iteration %d/%d..." % (i + 1, max_iters))

            idx = self.find_closest_centroids(centroids)

            if plot_progress:
                self.plot_progress_kmeans(centroids, previous_centroids, idx, K, i)
                previous_centroids = centroids

                print("Press enter to continue...")
                pause()


            centroids = self.compute_centroids(idx)

        if plot_progress:
            plt.hold(False)

        return centroids, idx

    def plot_progress_kmeans(self, centroids, previous, idx, K, i):

        if self.points != None:
            self.points.remove()

        self.points = plot_data_points(self.X, idx, K)
        plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=10, markeredgewidth=2)

        for j in range(len(centroids)):
            p1 = centroids[j, :]
            p2 = previous[j, :]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')

        plt.title('Iteration number %d' % (i + 1))
