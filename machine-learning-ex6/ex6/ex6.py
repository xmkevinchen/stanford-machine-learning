import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

def plot_data(X, y):

    positive = np.where(y == 1)
    negative = np.where(y == 0)

    plt.plot(X[positive, 0], X[positive, 1], 'k+', linewidth=1, markersize=7)
    plt.plot(X[negative, 0], X[negative, 1], 'ko', markerfacecolor='y', markersize=7)


def visualize_boundary(X, y, classifier):

    plot_data(X, y)
    x1_plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1_plot, x2_plot)
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contour(X1, X2, Z, 1, linecolor='b', linewidth=1)


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))

    sim = np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))
    return sim


print("Loading and Visualizing Data...\n")

data = sio.loadmat('ex6data1.mat')

X = data['X']
y = data['y'].ravel()

# plot_data(X, y)

print("\nTraining Linear SVM...")

classifier = svm.LinearSVC(C=1.0)
classifier.fit(X, y)

plt.figure()
visualize_boundary(X, y, classifier)

print("Evaluating the Gaussian Kernel...\n")

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussian_kernel(x1, x2, sigma)
print("Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :\n\t%.6f\n(this value should be about 0.324652)\n" % sim)

print("Loading and Visualizing Data...\n")

data = sio.loadmat('ex6data2.mat')
X = data['X']
y = data['y'].ravel()


print("Traing SVM with RBF Kernel (this may take 1 to 2 minutes)...\n")

C = 1
sigma = 0.1
gamma = 1 / (2 * sigma ** 2)
classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma)
classifier.fit(X, y)

plt.figure()
visualize_boundary(X, y, classifier)

print("Loading and Visualizing Data ...\n")

data = sio.loadmat('ex6data3.mat')
X = data['X']
y = data['y'].ravel()

Xval = data['Xval']
yval = data['yval'].ravel()

# plot_data(X, y)

from sklearn import grid_search

def dataset3_params(X, y):

    C_options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gamma_options = (1 / 2 * np.asarray(C_options) ** 2).tolist()

    clf = grid_search.GridSearchCV(svm.SVC(), {'C': C_options, 'gamma': gamma_options})
    clf.fit(X, y)

    best_params = clf.best_params_

    return best_params['C'], best_params['gamma']

def dataset3_params(X, y, Xval, yval):
    options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    error_min = np.inf

    C = 1
    gamma = 1 / 2 * 0.3 ** 2

    for C in options:
        for sigma in options:

            gamma = 1 / 2 * sigma ** 2

            clf = svm.SVC(C=C, gamma=gamma)
            clf.fit(X, y)
            predictions = clf.predict(Xval)

            error = np.mean(np.double(predictions != yval))

            if error < error_min:
                C_final = C
                gamma_final = gamma
                error_min = error
                print("new min error found with C, gamma = [%.2f, %f] with error = %f" % (C_final, gamma_final, error_min))

    C= C_final
    gamma = gamma_final

    print("The best value C, gamma = [%f, %f] with prediction error = %f" % (C, gamma, error_min))

    return C, gamma



C, gamma = dataset3_params(X, y, Xval, yval)
print("C, gamma = [%f, %f]" % (C, gamma))
classifier = svm.SVC(C=1, gamma=gamma)
classifier.fit(X, y)
plt.figure()
visualize_boundary(X, y, classifier)
