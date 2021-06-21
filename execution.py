import numpy as np
from brownian import BrownianMotion
from activebrownianparticles import Vicsek
import tij
import ballisticwithstop as bws
import matplotlib.pyplot as plt
import tkinter as tk
from animate import MovementAnimation as Ma

bsb = bws.BallStop(0.005, 200)
tij_array = bsb.total_movement()
timeline_array = tij.timeline(tij_array, 20)
quantities_ball_janus = tij.quantities_calculator(timeline_array)
tij.compare_quantities([quantities_ball_janus], ['Ball Stop Janus'], scale='linear')

'''
bsb = bws.BallStop(0.005, 20, 1, 500, 10000, 2000)
tij_array = bsb.total_movement()
timeline_array = tij.timeline(tij_array, 20)
quantities_ball_janus = tij.quantities_calculator(timeline_array)
tij.compare_quantities([quantities_ball_janus], ['Ball Stop Janus'], scale='log')
'''

'''
bsb = bws.BallStop(0.01, 20, 1, 100, 10000, 2000, brownian=False, janus=False)
tij_array = bsb.total_movement()
timeline_array = tij.timeline(tij_array, 20)
quantities_ball = tij.quantities_calculator(timeline_array)
'''
'''
data = quantities_ball_janus[0]
data = tij.regroup_data(data)
plt.plot(data[:, 0], data[:, 1]/sum(data[:,1]))

x = []
y = []
for i in range(70):
    x.append(20*(i+1))
    y.append((3/4)**i*1/4)
plt.plot(x, y)
plt.show()
'''
"""
'''
Brownian model
'''
br = BrownianMotion(20, 1, 1000, 10000, 2000)
brownian_tij = br.total_movement()
timeline_array = tij.timeline(brownian_tij, 20)
quantities_brown = tij.quantities_calculator(timeline_array)

'''
Ballistic with stop low density (100 * pi / 10000 * 100 = 3.14%)
'''


bs = bws.BallStop(1/200, 20, 1, 1000, 10000, 2000)
ballistic_tij = bs.total_movement()
timeline_array = tij.timeline(ballistic_tij, 20)
quantities_ball = tij.quantities_calculator(timeline_array)


'''
Ballistic with stop and Brownian low density (100 * pi / 10000 * 100 = 3.14%)
'''
bsb = bws.BallStop(1/200, 20, 1, 500, 10000, 2000, brownian=True)
ball_brownian_tij = bsb.total_movement()
timeline_array = tij.timeline(ball_brownian_tij, 20)
quantities_ball_brown = tij.quantities_calculator(timeline_array)


'''
Vicsek model
'''
vi = Vicsek(0.05, 20, 1, 500, 10000, 2000, 1, stop=False)
vi_tij = vi.total_movement()
timeline_array = tij.timeline(vi_tij, 20)
quantities_vicsek = tij.quantities_calculator(timeline_array)

'''
Vicsek model with stop
'''
vi = Vicsek(1.5/200, 20, 1, 500, 10000, 2000, 1, stop=True)
vi_tij = vi.total_movement()
timeline_array = tij.timeline(vi_tij, 20)
quantities_vicsek_stop = tij.quantities_calculator(timeline_array)

'''
Real data
'''
'''
pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_ICCSS17.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_conf = tij.quantities_calculator(timeline_array)

pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_InVS15.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_inv = tij.quantities_calculator(timeline_array)

pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_LH10.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_lh = tij.quantities_calculator(timeline_array)

pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_Thiers13.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_hosp = tij.quantities_calculator(timeline_array)
'''
pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_InVS15_1.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_in1 = tij.quantities_calculator(timeline_array)

pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_InVS15_2.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_in2 = tij.quantities_calculator(timeline_array)

pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_conf1.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_conf1 = tij.quantities_calculator(timeline_array)

pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_conf2.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
quantities_conf2 = tij.quantities_calculator(timeline_array)

quantities_array = [quantities_brown, quantities_conf1, quantities_conf2, quantities_in1, quantities_in2]
label_array = ['brownian', 'conf1', 'conf2', 'in1', 'in2']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_ball, quantities_conf1, quantities_conf2]
label_array = ['ball', 'conf1', 'conf2']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_ball_brown, quantities_conf1, quantities_conf2]
label_array = ['ball-brown', 'conf1', 'conf2']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_vicsek, quantities_conf1, quantities_conf2]
label_array = ['vicsek', 'conf1', 'conf2']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_vicsek_stop, quantities_conf1, quantities_conf2]
label_array = ['vicsek_stop', 'conf1', 'conf2']
tij.compare_quantities(quantities_array, label_array, scale='log')

'''
Execution
'''
'''
quantities_array = [quantities_brown, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['brownian', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_ball, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['ballistic', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_ball_brown, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['ball-brown', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_vicsek, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['vicsek', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='log')

quantities_array = [quantities_vicsek_stop, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['vicsek-stop', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='log')
'''

tij.make_hist(quantities_ball, "Ballistic motion with stop", scale='log')
tij.make_hist(quantities_ball_brown, "Ballistic and Brownian motion with stop", scale='semi_log')
tij.make_hist(quantities_brown, "Brownian motion", scale='semi_log')
tij.make_hist(quantities_vicsek, "Vicsek motion", scale='semi_log')
tij.make_hist(quantities_vicsek_stop, "Vicsek motion with stop", scale='semi_log')


quantities_array = [quantities_brown, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['brownian', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='semi_log')

quantities_array = [quantities_ball, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['ballistic', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='semi_log')

quantities_array = [quantities_ball_brown, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['ball-brown', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='semi_log')

quantities_array = [quantities_vicsek, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['vicsek', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='semi_log')

quantities_array = [quantities_vicsek_stop, quantities_conf, quantities_inv, quantities_lh, quantities_hosp]
label_array = ['vicsek-stop', 'conf', 'inv', 'lh', 'hosp']
tij.compare_quantities(quantities_array, label_array, scale='semi_log')

"""
