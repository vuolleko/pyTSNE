"""
t-Distributed Stochastic Neighbor Embedding
( https://lvdmaaten.github.io/tsne/ )

No sanity checks implemented.

Henri Vuollekoski, 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from time import time


class TSNE(object):
    """
    Implements basic t-Distributed Stochastic Neighbor Embedding.
    """

    def __init__(self, n_components=2, max_iter=1000, learning_rate=200.,
        momentum=0.5, momentum_final=0.8, early_exaggeration=4., n_early=250,
        init_method='pca', perplexity=50, perplex_tol=1e-4, perplex_evals_max=50,
        min_grad_norm2=1e-14, cost_min_since_max=30):
        """
        Set initial parameters for t-SNE.
        Input:
        - n_components: dimensionality of visualization
        - max_iter: max number of iterations
        - learning_rate: for gradient descent
        - momentum: for gradient descent
        - momentum_final: momentum after n_early iterations
        - early exaggeration: factor for affinities in high dimension
        - n_early: apply the above until this
        - init_method: initialization method 'pca' or 'rnorm'
        - perplexity: entropy requirement for bandwidth
        - perplex_tol: tolerance for error in evaluated entropy
        - perplex_evals_max: max number of tries for ok bandwidth
        - min_grad_norm2: abort if squared norm of gradient below this
        - cost_min_since_max: abort if no progress these iterations
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_final = momentum_final
        self.early_exaggeration = early_exaggeration
        self.n_early = n_early
        self.init_method = init_method
        self.perplexity = perplexity
        self.perplex_tol = perplex_tol
        self.perplex_evals_max = perplex_evals_max
        self.min_grad_norm2 = min_grad_norm2
        self.cost_min_since_max = cost_min_since_max

    def fit(self, data, animate=False, labels=None, anim_file="tsne_movie.mp4"):
        """
        Apply the t-SNE to input data.
        If animate=True, labels must be provided.
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

        # animate
        if animate and labels is not None:
            print("Recording animation.")
            writer = anim.writers['ffmpeg'](fps=10)
            fig, ax = plt.subplots()
            markers = self.plot_embedding2D(labels, ax)

            with writer.saving(fig, anim_file, 160):
                self._iterate(markers=markers, writer=writer)

        # don't animate
        else:
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
            dist2 = np.sum( (self.data - self.data[ii])**2., axis=1 )

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

        affin.flat[::self.n_samples+1] = 1.e-12  # set p_ii ~= 0
        affin = np.where(affin < 1.e-12, 1.e-12, affin)
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
        student.flat[::self.n_samples+1] = 1.e-12  # set q_ii ~= 0
        student = np.where(student < 1.e-12, 1.e-12, student)
        self.affin_ld = student / np.sum( student )

        self.gradient = 4. * np.sum(
                        ( (self.affin_hd - self.affin_ld) * student )[:, :, np.newaxis]
                        * (self.coord - self.coord[:, np.newaxis])
                                    , axis=1)

    def _iterate(self, markers=None, writer=None):
        """
        Iterate using gradient descent.
        """
        print("Iterating t-SNE...")

        coord_diff = np.zeros_like(self.coord)
        stepsize = np.ones_like(self.coord) * self.learning_rate

        print_period = 10
        ii = 0
        grad_norm2 = 1.
        cost_min = 1e99
        cost_min_since = 0
        costP = np.sum(self.affin_hd * np.log(self.affin_hd))
        time0 = time()
        while ii < self.max_iter and grad_norm2 > self.min_grad_norm2:
            if ii > 0 and ii%print_period==0:
                print( "{} iterations done. Time elapsed for last {}: "
                       "{:.2f} s. Gradient norm {:f}."
                       .format(ii, print_period, time() - time0, np.sqrt(grad_norm2)) )
                time0 = time()

            if ii == self.n_early:
                self.momentum = self.momentum_final
                self.affin_hd /= self.early_exaggeration  # cease "early exaggeration"

            self._set_gradient()

            # abort if no progress for a while
            cost = costP - np.sum( self.affin_hd * np.log(self.affin_ld) )
            if cost < cost_min:
                cost_min = cost
                cost_min_since = 0
            else:
                cost_min_since += 1
                if cost_min_since > self.cost_min_since_max:
                    print( "No progress for {} iterations. Aborting."
                           .format(cost_min_since) )
                    break

            # Decrease stepsize if previous step and current gradient in the same direction.
            # Otherwise increase stepsize (note the negative definition of gradient here).
            stepsize = np.where( (self.gradient > 0) == (coord_diff > 0),
                                np.maximum(stepsize * 0.8, 0.01 * self.learning_rate),
                                stepsize + 0.2 * self.learning_rate)

            coord_diff = self.learning_rate * self.gradient \
                         + self.momentum * coord_diff
            self.coord += coord_diff

            # update animation
            if writer:
                print("Animating iteration {}".format(ii))
                data = self._get_normalized_coords()
                for jj, text_artist in enumerate(markers):  # SLOW!
                    text_artist.set_x(data[jj, 0])
                    text_artist.set_y(data[jj, 1])
                writer.grab_frame()

            grad_norm2 = np.sum (self.gradient**2.)
            ii += 1

    def plot_embedding2D(self, labels, ax):
        """
        Plot the 2D data with labels.
        """
        n_class = 1. * len(np.unique(labels))

        markers = []  # list to hold text artists
        data = self._get_normalized_coords()

        # plot a number colored according to label
        for ii in xrange(self.n_samples):
            text_artist = ax.text(data[ii, 0], data[ii, 1], str(labels[ii]),
                                  color=plt.cm.Set1(labels[ii] / n_class),
                                  fontdict={'weight': 'bold', 'size': 12})
            markers.append(text_artist)

        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        # ax.set_xlim([data[:,0].min(), data[:,0].max()])
        # ax.set_ylim([data[:,1].min(), data[:,1].max()])

        return markers

    def _get_normalized_coords(self):
        """
        Return normalized coordinates for plotting.
        """
        data = self.coord.copy()
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - data_min) / (data_max - data_min)
        return data
