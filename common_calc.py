import numpy as np


def find_couples(neighbors, t):
    """
    This function finds all the couples at time t. The couples are all particles i and j that are neighbors at
    a particular time.

    :param neighbors: Array of all the neighbors of particle i
    :type neighbors: np.array of lists
    :param t: time of the iteration. It is equal to step * dt.
    :type t: float or int
    """
    new_couples = []
    for i, elt in enumerate(neighbors):
        if len(elt) > 1:
            new_couples_i = [(t, i, part) for part in elt if part > i]
            new_couples.extend(new_couples_i)
    return new_couples


def projection(vector, velocity_i, velocity_j):
    """
    This function returns True if velocity_i and velocity_j face opposite directions considering the vector that links
    the center of particle i and particle j.
    :param vector: vector that links the center of particle i and particle j.
    :type vector: np.array
    :param velocity_i: vector of the velocity of particle i
    :type velocity_i: np.array
    :param velocity_j: vector of the velocity of particle j
    :type velocity_j: np.array
    :return: True if the two velocity vectors face opposite directions
    :rtype: bool
    """
    a = np.dot(vector, velocity_i)
    b = np.dot(vector, velocity_j)
    return a <= 0 <= b
