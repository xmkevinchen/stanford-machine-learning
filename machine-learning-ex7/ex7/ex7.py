import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *
import scipy.io as sio

from kmeans import KMeans


plt.close('all')

X = sio.loadmat('ex7data2.mat')['X']

classifier = KMeans(X)

initial_centroids = np.asarray([[3, 3], [6, 2], [8, 5]])
idx = classifier.find_closest_centroids(initial_centroids)
print("Closest centroids for the first 3 examples:")
print(np.str(idx[0:3]))
print("(the closest centroids should be 0, 2, 1 respectively)")

centroids = classifier.compute_centroids(idx)
print("Centroids computed after initial finding of closest centroids: \n")
print(np.str(centroids))
print('(the centroids should be');
print('   [ 2.428301 3.157924 ]');
print('   [ 5.813503 2.633656 ]');
print('   [ 7.119387 3.616684 ]\n');

centroids, idx = classifier.run(plot_progress=True)
plt.show()
print("K-Means Done.")


print("Running K-Means clustering on pixels from an image. \n")

# Load an image of a bird
pixels = np.double(imread('bird_small.png'))

pixels = pixels / 255 # Divide by 255 so that all values are in the range 0-1

# Size of the image
img_shape = pixels.shape

'''
Reshape the image into a N x 3 matrix where N = number of pixels.
Each row will contain the Red, Green and Blue pixel values
This gives us our dataset matrix X that we will use K-Means on.
'''
X = pixels.reshape(img_shape[0] * img_shape[1], 3)

'''
Image Compression
To use the clusters of K-Means to compress an images
    1. Find the closest clusters for each example.
    2. Apply K-Means to compress an image
'''
classifier = KMeans(X, K=16)

centroids, idx = classifier.run()

# idx = classifier.find_closest_centroids(centroids)
X_recovered = centroids[idx, :]
X_recovered = X_recovered.reshape(img_shape)

plt.figure(2)
plt.subplot(121)
plt.imshow(pixels)
plt.title('Original')

plt.subplot(122)
plt.imshow(X_recovered)
plt.title("Compressed, with %d colors" % classifier.K)

plt.show()
