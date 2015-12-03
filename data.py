"""
Some datasets.

MNIST: http://yann.lecun.com/exdb/mnist/

Henri Vuollekoski, 2015
"""

import numpy as np
import struct


def read_MNIST(filename_images, filename_labels):
    """
    Return MNIST data set from files in IDX format.
    """
    pixels = _read_images(filename_images)
    labels = _read_labels(filename_labels)
    return [pixels, labels]


def _read_images(filename):
    """
    Read images in MNIST data set from file in IDX format.
    """
    with open(filename, 'rb') as fin:
        fin.seek(4)  # skip "magic number"
        n_samples = struct.unpack('>i', fin.read(4))[0]
        n_rows = struct.unpack('>i', fin.read(4))[0]
        n_cols = struct.unpack('>i', fin.read(4))[0]
        n_dim = n_cols * n_rows

        # data starts from byte 16
        pixels = np.fromfile(fin, dtype=np.ubyte)

    pixels = pixels.reshape(n_samples, n_dim)
    return pixels


def _read_labels(filename):
    """
    Read labels in MNIST data set from file in IDX format.
    """
    with open(filename, 'rb') as fin:
        fin.seek(8)  # skip "magic number" and number of labels
        # data starts from byte 8
        labels = np.fromfile(fin, dtype=np.ubyte)

    return labels
