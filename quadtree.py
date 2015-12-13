"""
Quadtree for Barnes-Hut t-SNE.

Henri Vuollekoski, 2015
"""

import numpy as np


class Quadnode(object):
    """
    Implements a Quadtree that stores 2D data. Each node holds the number of
    points, their "center of mass" and 4 child nodes.
    """

    def __init__(self, data, xx, yy, width):
        """
        Creates a node in a Quadtree.
        Inputs:
        - data: array of size (n,2)
        - coordinates xx, yy of node center
        - width (=height) of node
        """

        self.n_points = data.shape[0]
        self.width = width

        if self.n_points == 0:  # empty node
            self.center_of_mass = None
            return

        self.center_of_mass = np.mean(data, axis=0)

        if self.n_points > 1:

            # find appropriate quadrant for each data point
            north = data[:, 0] > yy
            south = np.invert(north)
            east = data[:, 1] > xx
            west = np.invert(east)

            # shift centers by 1/4 of current width
            shift = width / 4.
            width /= 2.

            # distribute points to quadrants
            self.northeast = Quadnode( data[north & east, :], xx+shift, yy+shift, width)
            self.southeast = Quadnode( data[south & east, :], xx+shift, yy-shift, width)
            self.southwest = Quadnode( data[south & west, :], xx-shift, yy-shift, width)
            self.northwest = Quadnode( data[north & west, :], xx-shift, yy+shift, width)

    def print_node(self, indent=0):
        """
        Prints a simple representation of the tree.
        """
        if self.n_points == 0:
            return

        indent_str = "   " * indent
        print(indent_str + "N={}, yx={}, width={}".format(self.n_points, self.center_of_mass, self.width))

        if self.n_points > 1:
            for node in [self.northeast, self.southeast, self.southwest, self.northwest]:
                node.print_node(indent+1)

    def summarize(self, coord, threshold):
        """
        Estimate "repulsive" contributions to gradient according to Barnes-Hut condition as in:
        L. van der Maaten, JML 15, 2014.
        F_rep = F_rep * Z / Z
        """
        if self.n_points == 0:
            return [0., 0.]

        dist2 = np.sum( (coord - self.center_of_mass)**2. )

        # no contribution from point itself
        if dist2 == 0.:
            return [0., 0.]

        # check whether this node can summarize its children (Barnes and Hut, 1986)
        if self.n_points == 1 or self.width / dist2 < threshold:
            student = 1. / (1. + dist2)
            denominator = self.n_points * student  # Z
            numerator = denominator * student * (coord - self.center_of_mass)  # F_rep * Z

        else:
            numerator = 0.
            denominator = 0.
            for node in [self.northeast, self.southeast, self.southwest, self.northwest]:
                summary = node.summarize(coord, threshold)
                numerator += summary[0]
                denominator += summary[1]

        return [numerator, denominator]
