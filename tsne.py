"""
t-Distributed Stochastic Neighbor Embedding
( https://lvdmaaten.github.io/tsne/ )

No sanity checks implemented.

Henri Vuollekoski, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

from time import time


class TSNE(object):
    """
    Implements basic t-Distributed Stochastic Neighbor Embedding.
    """

    def __init__(self, n_components=2, max_iter=10000, learning_rate=1000.,
        momentum=0.3, early_exaggeration=4., n_early=100, init_method='pca',
        perplexity=30, perplex_tol=1e-4, perplex_evals_max=50):
        """
        Set initial parameters for t-SNE.
        Input:
        - n_components: dimensionality of visualization
        - max_iter: max number of iterations
        - learning_rate: for gradient descent
        - momentum: for gradient descent
        - early exaggeration: factor for affinities in high dimension
        - n_early: apply the previous until this
        - init_method: initialization method 'pca' or 'rnorm'
        - perplexity: entropy requirement for bandwidth
        - perplex_tol: tolerance for error in evaluated entropy
        - perplex_evals_max: max number of tries for ok bandwidth
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.early_exaggeration = early_exaggeration
        self.n_early = n_early
        self.init_method = init_method
        self.perplexity = perplexity
        self.perplex_tol = perplex_tol
        self.perplex_evals_max = perplex_evals_max

    def fit(self, data, animate=False):
        """
        Apply the t-SNE.
        """
        self.data = data  # high-dimensional input data
        self.n_samples = data.shape[0]
        # self.n_dim = data.shape[1]

        # evaluate p_ij
        time0 = time()
        print("Calculating affinities in high dimension...")
        self._set_affin_hd()
        print("Done. Time elapsed {:.2f} s".format(time() - time0))

        # apply early exaggeration
        self.affin_hd *= self.early_exaggeration

        # initialize visualization
        time0 = time()
        if self.init_method == 'pca':
            print("Calculating PCA as initial guess...")
            self.coord = self._get_pca2()

        elif self.init_method == 'rnorm':
            print("Sampling initial distribution...")
            self.coord = np.random.randn(self.n_samples, self.n_components) * 1e-4

        print("Done. Time elapsed {:.2f} s".format(time() - time0))

        self._iterate()



    def _set_affin_hd(self):
        """
        Calculate pairwise affinities in high dimensional data.
        """
        affin = np.empty((self.n_samples, self.n_samples))
        log_perplexity = np.log2(self.perplexity)
        sigma22 = 1.e5  # initial guess for 2*sigma^2

        # calculate conditional probabilities
        for ii in xrange(self.n_samples):
            # print ii
            # all squared distances from ii
            dist2 = np.sum( (self.data - self.data[ii])**2., axis=1)

            s_min = 0.
            s_max = 1.e15
            error = self.perplex_tol + 1.  # just set big enough
            evals = 0

            # use binary search for sigma to get to desired perplexity
            while abs(error) > self.perplex_tol and evals < self.perplex_evals_max:

                # calculate affinities for current sigma
                denom = np.sum( np.exp( -dist2 / sigma22 ) ) - 1.
                affin[ii, :] = np.exp( -dist2 / sigma22 ) / denom

                # Shannon entropy = log2(perplexity)
                shannon = -np.sum( np.where(affin[ii, :] > 0.,
                          affin[ii, :] * np.log2(affin[ii, :])
                                  , 0.) )
                error = shannon - log_perplexity

                # P and Shannon entropy increase as sigma increases
                if error > 0:  # -> sigma too large
                    s_max = sigma22
                    sigma22 = (sigma22 + s_min) / 2.
                else:  # -> sigma too small
                    s_min = sigma22
                    sigma22 = (sigma22 + s_max) / 2.

                evals += 1

        affin.flat[::self.n_samples+1] = 0.  # set p_ii = 0
        affin = (affin + affin.T) / (2. * self.n_samples)  # make symmetric

        self.affin_hd = affin


    def _get_pca2(self):
        """
        Return the first principal components.
        """
        # contruct covariance matrix
        covmat = np.cov(self.data, rowvar=True)
        # get eigenvalues and eigenvectors
        eigval, eigvec = np.linalg.eigh(covmat)
        # sort eigenvectors (ascending) and pick two highest
        ind2 = np.argsort(eigval)[-self.n_components:]

        return eigvec[:, ind2]


    def _set_gradient(self):
        """
        Calculate pairwise affinities in 2D data using
        Student's t-distribution with 1 degree of freedom.

        Then calculate the gradient with respect to each 2D point.
        """

        # all pairwise distances
        dist2 = np.sum( (self.coord - self.coord[:, np.newaxis])**2., axis=2)

        student = 1. / (1. + dist2)
        student.flat[::self.n_samples+1] = 0.  # set q_ii = 0
        affin_ld = student / np.sum ( student )

        self.gradient = 4. * np.sum(
                        ( (self.affin_hd - affin_ld) * student )[:, :, np.newaxis]
                        * (self.coord - self.coord[:, np.newaxis])
                                    , axis=1)



    def _iterate(self):
        print("Iterating t-SNE...")

        coord_old = np.zeros_like(self.coord)
        ii = 0
        norm_grad2 = 1.
        time0 = time()
        while ii < self.max_iter and norm_grad2 > 1.e-14:
            if ii > 0 and ii%50==0:
                print("{} iterations done. Time elapsed for last 50: {:.2f} s. Gradient norm {:f}.".format(ii, time() - time0, np.sqrt(norm_grad2)))
                time0 = time()

            if ii == self.n_early:
                self.affin_hd /= self.early_exaggeration  # cease "early exaggeration"

            self._set_gradient()
            coord_diff = self.coord - coord_old
            coord_old = self.coord.copy()
            self.coord += self.learning_rate * self.gradient \
                          + self.momentum * coord_diff

            norm_grad2 = np.sum (self.gradient**2.)
            ii += 1

    def plot_embedding2D(self, labels, ax):
        """
        Plot the 2D data.
        """
        data = self.coord.copy()

        # normalize data to [0,1]
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - data_min) / (data_max - data_min)

        # plot a number colored according to label
        for ii in xrange(data.shape[0]):
            ax.text(data[ii, 0], data[ii, 1], str(labels[ii]),
                color=plt.cm.Set1(labels[ii] / 10.),
                fontdict={'weight': 'bold', 'size': 12})
        ax.set_xticks([])
        ax.set_yticks([])
