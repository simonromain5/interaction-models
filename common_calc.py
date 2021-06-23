import numpy as np
import tij


def find_couples(neighbors, t, velocites=None, janus=False):
    """
    This function finds all the couples at time t. The couples are all particles i and j that are neighbors at
    a particular time.

    :param neighbors: Array of all the neighbors of particle i
    :type neighbors: np.array of lists
    :param t: time of the iteration. It is equal to step * dt.
    :type t: float or int
    """
    new_couples = []

    if janus:
        new_couples_i = [(t, i, j) for j in elt if (j > i and np.dot(velocites[i], velocites[j]) <= 0)]

    else:
        new_couples_i = 0

    return new_couples


def projection(centers_array, velocity_i_array, velocity_j_array):
    """
    This function returns True if velocity_i and velocity_j face opposite directions considering the vector that links
    the center of particle i and particle j.
    :param centers_array: all the vectors that link the centers of particle i and particle j.
    :type centers_array: np.array
    :param velocity_i_array: vector of the velocity of particle i
    :type velocity_i_array: np.array
    :param velocity_j_array: array of vectors of the velocity of particle j (neighbors of i)
    :type velocity_j_array: np.array
    :return: True if the two velocity vectors face opposite directions
    :rtype: bool
    """
    a = np.einsum('ij,ij->i', centers_array, velocity_i_array)
    b = np.einsum('ij,ij->i', centers_array, velocity_j_array)
    return np.logical_and(a <= 0, b >= 0)


def cost_function(v, cl):
    mod = cl.Vicsek(v, 20, 1, 100, 10000, 2000, 1, stop=True)
    vi_tij = mod.total_movement()
    timeline_array = tij.timeline(vi_tij, 20)
    quantities_mod = tij.quantities_calculator(timeline_array)
    pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_conf1.dat'
    tij_array = tij.conversion(pt)
    timeline_array = tij.timeline(tij_array, 20)
    quantities_conf1 = tij.quantities_calculator(timeline_array)
    mse = 0
    for i, data1 in enumerate(quantities_mod[:-1]):
        data2 = quantities_conf1[i]
        min_data, max_data = max(min(data1), min(data2)), min(max(data1), max(data2))
        counts1, bins1 = np.histogram(data1, bins=np.logspace(np.log10(min_data), np.log10(max_data)), density=True)
        counts2, bins2 = np.histogram(data2, bins=np.logspace(np.log10(min_data), np.log10(max_data)), density=True)
        non_null_index = np.where(np.logical_and(counts1 != 0, counts2 != 0))
        counts1, counts2 = np.log10(counts1[non_null_index]), np.log10(counts2[non_null_index])
        mse += np.mean((counts1 - counts2) ** 2)

    return np.sqrt(mse)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_index_sup1(neighbors):

    def len_elt_sup1(elt):
        return len(elt) > 1

    truth_array = np.vectorize(len_elt_sup1)(neighbors)
    return np.where(truth_array)[0]
