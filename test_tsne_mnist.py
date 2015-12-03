"""
Apply t-Distributed Stochastic Neighbor Embedding to MNIST digits.

Henri Vuollekoski, 2015
"""

import data
import matplotlib.pyplot as plt

from sklearn import datasets
from time import time

import tsne


time0 = time()
print("Loading data...")

# Dataset of 28x28 images
# filename_train_images = "MNIST/t10k-images.idx3-ubyte"
# filename_train_labels = "MNIST/t10k-labels.idx1-ubyte"
# images_train, labels_train = data.read_MNIST(filename_train_images, filename_train_labels)
# limit data...
# n_samples = 2000
# images_train = images_train[:n_samples]
# labels_train = labels_train[:n_samples]

# Dataset of 8x8 images
digits = datasets.load_digits(n_class=10)
images_train = digits.data
labels_train = digits.target

print("Done. Time elapsed {:.2f} s".format(time() - time0))

vis = tsne.TSNE(max_iter=500)
vis.fit(images_train)

fig, ax = plt.subplots()
vis.plot_embedding2D(labels_train, ax)
plt.show()
