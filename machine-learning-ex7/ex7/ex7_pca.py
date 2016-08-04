#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import scipy.io as sio
import scipy.misc
from kmeans import *

def draw_line(start, end, *args, **kwargs):

    x, y = zip(start, end)
    ax = plt.gca()
    ax.plot(x, y, *args, **kwargs)


def display_data(X, width=None):

    m, n = X.shape

    if width == None:
        width = int(round(n ** .5))

    height = int(n / width)

    rows = int(np.floor(m ** .5))
    cols = int(np.ceil(m / rows))

    pad = 1

    # setup blank display


    Z = -np.ones((pad + rows * (height + pad), (pad + cols * (width + pad))))

    counter = 0

    for j in range(0, rows):
        for i in range(0, cols):
            y = j * (height + pad) + pad
            x = i * (width + pad) + pad

            max_val = np.max(abs(X[counter, :]))

            Z[y:y + height, x:x + width] = X[counter, :].reshape(height, width, order='F') / max_val
            counter += 1

    ax = plt.gca()
    fig = plt.gcf()
    img = ax.imshow(Z, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    return img, Z


print("Visualizing example dataset for PCA.\n")

X = sio.loadmat('ex7data1.mat')['X']

plt.close('all')

plt.figure(1)
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])
plt.axis('square')
plt.show(False)


print("Running PCA on example dataset.\n")

classifier = KMeans(X)

# Before running PCA, it's important to first normalize X
(X_norm, mu, sigma) = feature_normalize(X)

# Run PCA
(U, S) = pca(X_norm)

plt.hold(True)
draw_line(mu, mu + 1.5 * S[0, 0] * U[:, 0].T, 'k-', linewidth=2)
draw_line(mu, mu + 1.5 * S[1, 1] * U[:, 1].T, 'k-', linewidth=2)
plt.hold(False)

print("Top eigenvector: ")
print(" U[:, 0] = %f %f " %(U[0, 0], U[1, 0]))
print("\n (you should expect to see -0.707107, -0.707107)\n")

print('Program paused. Press enter to continue.')
pause()

'''
=================== Part 3: Dimension Reduction ===================
You should now implement the projection step to map the data onto the
first k eigenvectors. The code will then plot the data in this reduced
dimensional space.  This will show you what the data looks like when
using only the corresponding eigenvectors to reconstruct it.

'''
print("Dimension reduction on example dataset.\n")

plt.figure(2)
plt.show(False)
plt.hold(True)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4, 3, -4, 3])
plt.axis('square')

classifier.X = X_norm
K = 1
Z = project_data(X_norm, U, K)
print("Projection of the first example: %f" % Z[0]);
print('\n(this value should be about 1.481274)\n');

X_recovered = recover_data(Z, U, K)
print("Approximation of the first example: %f %f" % (X_recovered[0, 0], X_recovered[0, 1]))
print("\n(this value should be about -1.047419 -1.047419)")

plt.plot(X_recovered[:, 0], X_recovered[:, 1], 'ro')
for i in range(X_norm.shape[0]):
    draw_line(X_norm[i, :], X_recovered[i, :], '--k', linewidth=1)

plt.hold(False)

print('Program paused. Press enter to continue.\n');
pause()

'''
=============== Part 4: Loading and Visualizing Face Data =============
We start the exercise by first loading and visualizing the dataset.
The following code will load the dataset into your environment

'''

print("Loading face dataset.\n")

X = sio.loadmat('ex7faces.mat')['X']
plt.figure()
data = display_data(X[range(100), :])
plt.show(False)

print('Program paused. Press enter to continue.\n')
pause()

'''
=========== Part 5: PCA on Face Data: Eigenfaces  ===================
  Run PCA and visualize the eigenvectors which are in this case eigenfaces
  We display the first 36 eigenfaces.

'''
print("Running PCA on face dataset.\n(this might take a minute or two...)\n")

X_norm, mu, sigma = feature_normalize(X)
U, S = pca(X_norm)

plt.figure()
display_data(U[:, 0:36].T)
plt.show(False)

print('Program paused. Press enter to continue.\n')
pause()

'''
============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors
%  If you are applying a machine learning algorithm
'''

print('\nDimension reduction for face dataset.\n')
K = 100
Z = project_data(X_norm, U, K)

print("The projected data Z has a size of: %f  %f" %(Z.shape[0], Z.shape[1]))


'''
==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed
'''

X_rec = recover_data(Z, U, K)

plt.figure()
plt.subplot(121)
display_data(X_norm[0:100, :])
plt.title("Original faces")
plt.axis('equal')

plt.subplot(122)
display_data(X_rec[0:100, :])
plt.title('Recovered faces')
plt.axis('equal')
plt.show(False)

print('Program paused. Press enter to continue.\n')
pause()

'''
=== Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.
'''

plt.close('all')

from mpl_toolkits.mplot3d import Axes3D

A = np.double(scipy.misc.imread('bird_small.png'))
A = A / 255
img_size = A.shape

X = A.reshape((img_size[0] * img_size[1], 3), order='F')

K = 16
classifier = KMeans(X, K=K)
centroids, idx = classifier.run()

sel = (np.floor(np.random.rand(1000, 1) * X.shape[0]) + 1).astype(int)
sel = sel.ravel()

n = classifier.K
palette = cm.colors.hsv_to_rgb(np.c_[np.arange(n).reshape(-1, 1) / n, np.ones((n, 2))])
colors = palette[idx[sel], :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = X[sel, 0]
ys = X[sel, 1]
zs = X[sel, 2]
c = [1, 1, 1, 0]
points = ax.scatter(xs, ys, zs, s=50, c=c, edgecolors=colors)
ax.invert_xaxis()

plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships');
plt.show(False)

print('Program paused. Press enter to continue.\n')
pause()

X_norm, mu, sigma = feature_normalize(X)
U, S = pca(X_norm)
Z = project_data(X_norm, U, 2)

plt.figure()
plot_data_points(Z[sel, :], idx[sel], K)
plt.title("Pixel dataset plotted in 2D, using PCA for dimensionality reduction")
plt.show(False)

print('Program paused. Press enter to continue.\n')
pause()
