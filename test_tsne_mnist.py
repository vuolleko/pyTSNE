"""
Apply t-Distributed Stochastic Neighbor Embedding to MNIST digits.

Henri Vuollekoski, 2015
"""

from time import time

import data
import tsne
import bh_tsne


time0 = time()
print("Loading data...")

# Dataset of 28x28 images
filename_train_images = "MNIST/t10k-images.idx3-ubyte"
filename_train_labels = "MNIST/t10k-labels.idx1-ubyte"
images_train, labels_train = data.read_MNIST(filename_train_images, filename_train_labels)
n_samples = 500  # limit data...
images_train = images_train[:n_samples] / 255.
labels_train = labels_train[:n_samples]
images_train = tsne.get_pca_fit(images_train, 50)

print("Done. Time elapsed {:.2f} s".format(time() - time0))

anim = False

if anim:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# vis = tsne.TSNE(max_iter=500)
vis = bh_tsne.BH_TSNE(max_iter=500, bh_threshold=0.5)
vis.fit(images_train, animate=anim, labels=labels_train, anim_file="tsne_movie.mp4")

if not anim:
    fig, ax = plt.subplots()
    vis.plot_embedding2D(labels_train, ax)
    plt.show()
