import numpy as np
import tij


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
