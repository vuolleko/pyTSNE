"""
Apply t-Distributed Stochastic Neighbor Embedding to MNIST digits.
http://yann.lecun.com/exdb/mnist/index.html

Henri Vuollekoski, 2015
"""

from time import time

import data
import tsne
import bh_tsne

time00 = time()
time0 = time()
print("Loading data...")

# Dataset of 28x28 images
filename_train_images = "MNIST/train-images.idx3-ubyte"
filename_train_labels = "MNIST/train-labels.idx1-ubyte"
images_train, labels_train = data.read_MNIST(filename_train_images, filename_train_labels)

n_samples = 1000  # limit data...
images_train = images_train[:n_samples] / 255.
labels_train = labels_train[:n_samples]
images_train = tsne.get_pca_proj(images_train, 30)

print("Done. Time elapsed {:.2f} s".format(time() - time0))

anim = False

if anim:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

vis = tsne.TSNE(max_iter=1000)
# vis = bh_tsne.BH_TSNE(max_iter=500, bh_threshold=0.5)
vis.fit(images_train, animate=anim, labels=labels_train, anim_file="tsne_movie.mp4")

print("Total time: {:.2f} s".format(time()-time00))

if not anim:
    fig, ax = plt.subplots()
    vis.plot_embedding2D(labels_train, ax)
    plt.show()
