import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def conversion(path):
    """
    Converts a tij.dat file into a np.array.

    :param path: path of the tij.dat file
    :type path: str
    :return: np.array of the tij data
    :rtype: np.array
    """
    df = pd.read_csv(path, sep='\t')
    tij_array = df.to_numpy()
    return tij_array


def unique(ar):
    """
    This function gives each unique value of an array and the number of occurrence of the value

    :param ar: Array that is studied
    :type ar: np.array
    :return: Unique values of ar and the number of occurrences
    :rtype: tuple of np.array
    """
    values, counts = np.unique(ar, return_counts=True)
    return values, counts


def common(ar1, ar2):
    """
    This functions returns the common rows of ar1 and ar2
    :param ar1: First array
    :type ar1: np.array
    :param ar2: Second array
    :type ar2: np.array
    :return: array of common rows
    :rtype: np.array
    """
    common_array = np.array([x for x in set(tuple(x) for x in ar1) & set(tuple(x) for x in ar2)])
    return common_array


def lost(ar1, ar2):
    """
    This function finds the rows that are in ar1 but not in ar2. These rows are called the lost rows.
    :param ar1: First array
    :type ar1: np.array
    :param ar2: Second array
    :type ar2: np.array
    :return: array of the lost rows
    :rtype: np.array
    """

    set1 = {tuple(x) for x in ar1}
    set2 = {tuple(x) for x in ar2}
    lost_set = (set1 ^ set2) & set1
    if len(lost_set) != 0:
        lost_array = np.array(list(lost_set))
    else:
        lost_array = np.empty((0, 2), dtype=int)
    return lost_array


def new(ar1, ar2):
    """
    This function finds the rows that are in ar2 but not in ar1. These rows are called the new rows.
    :param ar1: First array
    :type ar1: np.array
    :param ar2: Second array
    :type ar2: np.array
    :return: array of the lost rows
    :rtype: np.array
    """
    set1 = {tuple(x) for x in ar1}
    set2 = {tuple(x) for x in ar2}
    new_set = set2 - set1
    if len(new_set) != 0:
        new_array = np.array(list(new_set))
    else:
        new_array = np.empty((0, 2), dtype=int)
    return new_array


def add_time(time, couples, timeline_array):
    """
    This function adds
    :param time:
    :param couples:
    :param timeline_array:
    :return:
    """
    for elt in couples:
        i = elt[0]
        j = elt[1]
        if i < j:
            timeline_array[i, j].append(time)
        else:
            timeline_array[j, i].append(time)
    return timeline_array


def timeline(tij_array, dt):
    """
    This function returns an array of timelines of interactions between all the particles. A timeline between particle
    i and j has the following form [t1, t2, t3, t4 ...] with all the odd elements the time of the beginning of an
    interaction and all the even elements the time of the end of an interaction. As the interaction between i and j is
    strictly the same as the interaction between j and i the array should be symmetric, with all the diagonal elements
    equal to 0 (no interaction between i and i). In our case the array is strictly upper triangular (no need to keep in
    memory all the elements).

    :param tij_array: Array of the tij elements, that are needed to create the timeline array
    :type tij_array: np.array
    :param dt: Increment of time for each step
    :type dt: float or int
    :return: Array of timelines.
    :rtype: np.array of lists
    """
    time_array, counts = unique(tij_array[:, 0])
    ij_array = tij_array[:, 1:]
    ij_array = np.int64(ij_array)
    i_min = np.min(ij_array)
    i_max = np.max(ij_array)
    ij_array = ij_array - i_min
    timeline_size = (i_max - i_min + 1, ) * 2
    timeline_array = np.frompyfunc(list, 0, 1)(np.empty(timeline_size, dtype=object))
    count = counts[0]
    couples = ij_array[0:count]
    old_time = time_array[0]
    timeline_array = add_time(old_time, couples, timeline_array)

    for step, time in enumerate(time_array[1:]):
        new_count = count + counts[step + 1]
        couples1 = ij_array[count: new_count, :]
        new_couples = new(couples, couples1)
        lost_couples = lost(couples, couples1)
        if new_couples.size > 0:
            timeline_array = add_time(time, new_couples, timeline_array)
        if lost_couples.size > 0:
            timeline_array = add_time(old_time + dt, lost_couples, timeline_array)
        couples = couples1
        count = new_count
        old_time = time
    return timeline_array


def quantities_calculator(timeline_array, dec=1):
    """
    Calculates 4 different quantities - contact time, inter-contact time, number of contacts and weight - that are
    needed to compare and validate different models with real data.

    :param timeline_array: Array of timelines.
    :type timeline_array: np.array of lists
    :param dec: decimals to which we around the quantities. Default is equal to 1
    :type dec: int, optional
    """
    contact_time_array = []
    inter_contact_time_array = []
    number_contact_array = []
    link_weight_array = []
    for elt in timeline_array:
        for elt1 in elt:
            if len(elt1) % 2 == 1:
                elt1.pop()
            if len(elt1) > 0:
                number_contact_array.append(len(elt1) // 2)
                contact_time = [b-a for a, b in tuple(zip(elt1, elt1[1:]))[::2]]
                contact_time_array.extend(contact_time)
                link_weight_array.append(sum(contact_time))
                inter_contact_time = [b-a for a, b in tuple(zip(elt1[1:], elt1[2:]))[::2]]
                inter_contact_time_array.extend(inter_contact_time)
    contact_time_array, inter_contact_time_array = np.array(contact_time_array), np.array(inter_contact_time_array)
    number_contact_array, link_weight_array = np.array(number_contact_array, dtype=int), np.array(link_weight_array)
    contact_time_array = np.around(contact_time_array, decimals=dec)
    inter_contact_time_array = np.around(inter_contact_time_array, decimals=dec)
    link_weight_array = np.around(link_weight_array, decimals=dec)
    return contact_time_array, inter_contact_time_array, number_contact_array, link_weight_array


def regroup_data(ar):
    """
    This function regroups the quantities with the same value and calculates the number of occurrence of the value.
    The results are then put in a array where for all i, the first element of row i is value i and the second element
    of row i is its number of occurrences.

    :param ar: Array of all the values, of shape (n, )
    :type ar: np.array
    :return: array of shape (n', 2) of values and counts
    :rtype: np.array
    """
    values, counts = unique(ar)
    return np.concatenate((values.reshape((-1, 1)), counts.reshape((-1, 1))), axis=1)


def representation(quantities, title, scale='linear'):
    """
    Represents 4 different quantities - contact time, inter-contact time, number of contacts and weight - in scatter
    plots.
    :param quantities: tuple of the 4 quantities that are represented
    :type quantities: tuple of np.arrays
    :param title: Title of the figure
    :type title: str
    :param scale: Scale of the plot. Can be 'linear' (default), 'log' or 'semi-log'
    :type scale: str, optional
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    index = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i, data in enumerate(quantities):
        a = index[i][0]
        b = index[i][1]
        new_data = regroup_data(data)
        x = new_data[:, 0]
        y = new_data[:, 1]
        axs[a, b].scatter(x, y)

        if i == 0:
            axs[a, b].set_xlabel('Contact duration')
            axs[a, b].set_ylabel('Distribution of contact duration')

        if i == 1:
            axs[a, b].set_xlabel('Inter-contact duration')
            axs[a, b].set_ylabel('Distribution of inter contact duration')

        if i == 2:
            axs[a, b].set_xlabel('Number of contacts')
            axs[a, b].set_ylabel('Distribution of number of contacts')

        if i == 3:
            axs[a, b].set_xlabel('Weight')
            axs[a, b].set_ylabel('Weight distribution')

        if scale != 'linear':
            if scale == 'log':
                axs[a, b].set_xscale('log')
                axs[a, b].set_yscale('log')
            elif scale == 'semi_log':
                axs[a, b].set_yscale('log')
        axs[a, b].grid()
    plt.show()


def make_hist(quantities, title, scale='linear'):
    """
    Represents 4 different quantities - contact time, inter-contact time, number of contacts and weight - in histograms.

    :param quantities: tuple of the 4 quantities that are represented
    :type quantities: tuple of np.arrays
    :param title: Title of the figure
    :type title: str
    :param scale: Scale of the plot. Can be 'linear' (default), 'log' or 'semi-log'
    :type scale: str, optional
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    index = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i, data in enumerate(quantities):
        a = index[i][0]
        b = index[i][1]

        if i == 0:
            axs[a, b].set_xlabel('Contact duration')
            axs[a, b].set_ylabel('Distribution of contact duration')

        if i == 1:
            axs[a, b].set_xlabel('Inter-contact duration')
            axs[a, b].set_ylabel('Distribution of inter contact duration')

        if i == 2:
            axs[a, b].set_xlabel('Number of contacts')
            axs[a, b].set_ylabel('Distribution of number of contacts')

        if i == 3:
            axs[a, b].set_xlabel('Weight')
            axs[a, b].set_ylabel('Weight distribution')

        if scale == 'linear':
            axs[a, b].hist(data, bins='auto')
        elif scale == 'log':
            axs[a, b].hist(data, bins=np.logspace(np.log10(min(data)), np.log10(max(data))), log=True, density=True)
            axs[a, b].set_xscale('log')
        elif scale == 'semi_log':
            axs[a, b].hist(data, bins='auto', log=True)
        axs[a, b].grid()
    plt.show()


def compare_quantities(quantities_array, label_array, title='Comparison tij data', scale='linear'):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    index = [[0, 0], [0, 1], [1, 0], [1, 1]]
    colors = 'bgrmcykw'
    markers = '*+x><d^8p'

    for i in range(4):
        a = index[i][0]
        b = index[i][1]

        if i == 0:
            axs[a, b].set_xlabel('Contact duration')
            axs[a, b].set_ylabel('Distribution of contact duration')

        if i == 1:
            axs[a, b].set_xlabel('Inter-contact duration')
            axs[a, b].set_ylabel('Distribution of inter contact duration')

        if i == 2:
            axs[a, b].set_xlabel('Number of contacts')
            axs[a, b].set_ylabel('Distribution of number of contacts')

        if i == 3:
            axs[a, b].set_xlabel('Weight')
            axs[a, b].set_ylabel('Weight distribution')

        axs[a, b].grid()

    for j, data in enumerate(quantities_array):
        data_label = label_array[j]

        for i in range(4):
            a = index[i][0]
            b = index[i][1]
            data = quantities_array[j][i]

            if scale == 'linear':
                counts, bins = np.histogram(data, bins='auto', density=True)

            elif scale == 'log':
                counts, bins = np.histogram(data, bins=np.logspace(np.log10(min(data)), np.log10(max(data))), density=True)
                axs[a, b].set_xscale('log')
                axs[a, b].set_yscale('log')

            elif scale == 'semi_log':
                counts, bins = np.histogram(data, bins='auto')
                axs[a, b].set_yscale('log')

            bins = np.array([(elt + bins[i+1])/2 for i, elt in enumerate(bins[:-1])])
            null_index = np.where(counts != 0)[0]
            bins, counts = bins[null_index], counts[null_index]

            if j == 0:
                axs[a, b].plot(bins, counts, c=colors[j], label=data_label)

            else:
                axs[a, b].plot(bins, counts, c=colors[j], marker=markers[j], markersize=3, label=data_label, linestyle='None')
            axs[a, b].legend()

    plt.show()
