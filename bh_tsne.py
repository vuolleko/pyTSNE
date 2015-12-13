"""
Barnes-Hut accelerated t-Distributed Stochastic Neighbor Embedding
( https://lvdmaaten.github.io/tsne/ )

No sanity checks implemented.

Henri Vuollekoski, 2015
"""

import numpy as np
# import scipy.sparse as sparse

from time import time

from vptree import VPnode
from quadtree import Quadnode
from tsne import TSNE


class BH_TSNE(TSNE):
    """
    Implements Barnes-Hut accelerated t-Distributed Stochastic Neighbor Embedding.
    """

    def __init__(self, bh_threshold=0.5, *args, **kwargs):
        """
        Set initial parameters for t-SNE.
        Input:
        - bh_threshold: threshold for Barnes-Hut condition
        See also parent class tsne.TSNE.
        """
        self.bh_threshold = bh_threshold
        super(BH_TSNE, self).__init__(*args, **kwargs)
        self.cost_min_since_max = 0  # not implemented for Barnes-Hut

    def _set_affin_hd(self):
        """
        Calculate pairwise affinities for nearest neighbors in
        high dimensional data.
        """
        # TODO use sparse matrices
        # affin = sparse.lil_matrix((self.n_samples, self.n_samples))
        affin = np.zeros((self.n_samples, self.n_samples))

        # construct a Vantage-point tree from data
        time0 = time()
        print("Constructing a Vantage-point tree...")
        root = VPnode(np.arange(self.n_samples), self.data)
        print("VP done. Time elapsed {:.2f} s".format(time() - time0))

        # number of neighbors to account for
        n_neighbors = 3 * self.perplexity
        distances = np.empty(n_neighbors)  # array to store distances

        # calculate conditional probabilities
        for ii in range(self.n_samples):
            neighbors = np.arange(n_neighbors)  # array to store indices
            distances[:] = np.inf  # init to infinity
            root.find_neighbors(ii, neighbors, distances, self.data)
            affin[ii, neighbors] = self._affin_bin_search_sigma(distances)

        affin = (affin + affin.T) / (2. * self.n_samples)  # make symmetric

        # self.affin_hd = affin.tocsr()
        self.affin_hd = affin

    def _set_gradient(self):
        """
        Estimate the gradient using a quadtree.
        """
        # construct a Quadtree from data
        yy, xx = np.median(self.coord, axis=0)
        width = np.max(self.coord) - np.min(self.coord)
        qdroot = Quadnode(self.coord, xx, yy, width * 2.)

        # inds = self.affin_hd.nonzero()
        # diffs = self.coord[inds[0], :] - self.coord[inds[1], :] #[:, np.newaxis]  #PROB

        diffs = self.coord - self.coord[:, np.newaxis]
        F_attr = np.sum( (self.affin_hd / (1. + np.sum(diffs**2., axis=2)) )[:, :, np.newaxis]
                         * diffs, axis=1)
        F_rep = np.empty_like(F_attr)
        student_sum = 0.
        for ii in range(self.n_samples):
            summary = qdroot.summarize(self.coord[ii, :], self.bh_threshold)
            F_rep[ii, :] = summary[0]
            student_sum += summary[1]

        F_rep /= student_sum
        self.gradient = 4. * (F_attr + F_rep)
