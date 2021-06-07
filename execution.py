from brownian import BrownianMotion
from activebrownianparticles import Vicsek
import tij
import ballisticwithstop as bws
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from animate import MovementAnimation as Ma


'''
Brownian model
'''
a = BrownianMotion(0.1, 1,  100, 10000, 5000)
brown_animation = Ma(a, 1000)
brownian_tij = a.total_movement()

title = 'Brownian motion'
timeline_array = tij.timeline(brownian_tij, 0.1)
quantities = tij.quantities_calculator(timeline_array)
tij.make_hist(quantities, title, scale='log')
tij.representation(quantities, title, scale='log')

'''
Ballistic with stop low density (100 * pi / 10000 * 100 = 3.14%)
'''


a = bws.BallStop(1, 0.1, 1, 100, 10000, 10000)
ballistic_animation = Ma(a, 1000)
ballistic_tij = a.total_movement()

timeline_array = tij.timeline(ballistic_tij, 0.1)
quantities = tij.quantities_calculator(timeline_array)
title = 'Ballistic with stop - density = ' + str(np.pi) + '%'
tij.make_hist(quantities, title, scale='semi_log')
tij.representation(quantities, title, scale='semi_log')

'''
Ballistic with stop and Brownian low density (100 * pi / 10000 * 100 = 3.14%)
'''
bsb = bws.BallStop(1, 0.1, 1, 100, 10000, 10000, brownian=True)
ballistic_brown_animation = Ma(bsb, 1000)
ball_brownian_tij = bsb.total_movement()

title = 'Ballistic with stop and Brownian motion'
timeline_array = tij.timeline(ball_brownian_tij, 0.1)
quantities = tij.quantities_calculator(timeline_array)
tij.make_hist(quantities, title, scale='log')
tij.representation(quantities, title, scale='log')

'''
Vicsek model - noise=4
'''
vi = Vicsek(1, 1.5, 0.1, 1, 100, 10000, 10000)
vicsek_animation = Ma(vi, 1000)
vi_tij = vi.movement()
title = 'Vicsek model'
timeline_array = tij.timeline(vi_tij, 0.1)
quantities = tij.quantities_calculator(timeline_array)
tij.make_hist(quantities, title, scale='log')
tij.representation(quantities, title, scale='log')

'''
Real data
'''
pt = '/home/romain/Documents/Stage_CPT/tij_data/tij_ICCSS17.dat'
tij_array = tij.conversion(pt)
timeline_array = tij.timeline(tij_array, 20)
title = 'Conference room'
quantities = tij.quantities_calculator(timeline_array)

tij.make_hist(quantities, title, scale='log')
tij.representation(quantities, title, scale='log')
