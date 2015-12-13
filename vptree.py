"""
Vantage-point tree.

Henri Vuollekoski, 2015
"""

import numpy as np


class VPnode(object):
    """
    Implements a Vantage-point tree that stores indices to data.
    Distance calculations use the squared L2 norm.
    """

    def __init__(self, inds, data):
        """
        Inputs:
        - inds: ordered array of indices
        - data: indexable data for distance calculation, in random order
        """

        if len(inds) == 0:  # empty node
            self.ind = None
            self.radius = None
            return

        self.ind = inds[0]  # the "center" of this node
        inds = inds[1:]

        if len(inds) > 0:

            # calculate all distances from ind using the squared L2 norm
            distances = np.sum( (data[1:, :] - data[0, :])**2., axis=1)
            data = data[1:, :]

            # set the node radius as the median distance from ind to others
            self.radius = np.median( distances )

            inds_inside = distances < self.radius
            inds_outside = np.invert( inds_inside )

            # attach children
            self.inside = VPnode(inds[inds_inside], data[inds_inside])
            self.outside = VPnode(inds[inds_outside], data[inds_outside])

        else:  # leaf node
            self.radius = None
            self.inside = VPnode([], [])
            self.outside = VPnode([], [])

    def print_node(self, indent=0):
        """
        Prints a simple representation of the tree.
        """
        if self.ind is None:
            return

        indent_str = "    " * indent
        print(indent_str + "* (Node {})".format(self.ind))
        if self.inside.ind is not None:
            print(indent_str + "  - Inside {}:".format(self.ind))
            self.inside.print_node(indent+1)
        if self.outside.ind is not None:
            print(indent_str + "  - Outside {}:".format(self.ind))
            self.outside.print_node(indent+1)

    def find_neighbors(self, ind, neighbors, distances, data):
        """
        Returns indices to k nearest neighbors of ind.
        Inputs:
        - neighbors: a numpy array of indices sized k
        - distances: a numpy array of distances sized k, should be initialized to inf
        - data: indexable data for distance calculation
        """

        if self.ind is None:  # empty node
            return

        # calculate distance from node to point
        dist = np.sum( (data[self.ind, :] - data[ind, :])**2. )

        # set current radius of search to max of current neighbor distances
        radius_search = distances[-1]

        if ind != self.ind and dist < radius_search:
        # assign the current node as the k+1 neighbor and sort according to distances
            distances[-1] = dist
            neighbors[-1] = self.ind
            ind_sort = np.argsort(distances)
            distances[:] = distances[ind_sort]
            neighbors[:] = neighbors[ind_sort]

        if self.radius is None:  # leaf-node
            return

        if dist >= radius_search + self.radius:  # target area completely outside
            self.outside.find_neighbors(ind, neighbors, distances, data)

        elif self.radius > radius_search + dist:  # target area completely inside
            self.inside.find_neighbors(ind, neighbors, distances, data)

        else:  # target area may be both inside and outside
            self.inside.find_neighbors(ind, neighbors, distances, data)
            self.outside.find_neighbors(ind, neighbors, distances, data)

        return neighbors
