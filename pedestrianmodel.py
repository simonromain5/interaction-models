import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize
import basemodel as bm
import matplotlib.pyplot as plt


class Pedestrian(bm.AbstractBwsAbpModel):

    def __init__(self, v, n_particles, dt=20, radius=1, surface=10000, n_steps=2000, janus=False, stop=False):
        super().__init__(v, n_particles, dt, radius, surface, n_steps, janus, stop)
        #self.position_array[0] = [97, 50]
        self.d_max = 10
        print(self.optimum_angle(np.pi/3))
        pairs_angle_array, pairs_dist_array = self.angle_dist(0)
        angle_array = np.linspace(0, 2/3*np.pi, 500)
        y = np.empty((500,))
        y1 = np.empty((500,))
        y2 = np.empty((500,))
        for i, angle in enumerate(angle_array):
            y[i] = self.collision_function(np.array([angle]), pairs_angle_array, pairs_dist_array)
            y1[i] = self.cost_angle_function(np.array([angle]), np.pi/3)
            y2[i] = 3 / np.cos(angle)

        plt.plot(angle_array, y, label='f(angle)')
        plt.plot(angle_array, y1, label='d(angle)')
        plt.legend()
        #plt.plot(angle_array, y2)
        plt.xlabel("angle")
        plt.ylabel("distance")
        #plt.ylim(0, 11)
        plt.show()

    def collision_function(self, angle, pairs_angle_array, pairs_dist_array):

        view_index = np.where(np.logical_and(pairs_angle_array > angle - 0.1, pairs_angle_array < angle + 0.1))[0]
        if view_index.size == 0:
            position_i = self.position_array[0]
            direction = np.array([np.cos(angle), np.sin(angle)])
            border_vector = position_i + self.d_max * np.array([np.cos(angle)[0], np.sin(angle)[0]])
            max_index = np.where(border_vector > self.side)[0]
            min_index = np.where((border_vector < 0))[0]
            out = self.d_max

            if max_index.size > 0:
                out = optimize.minimize(self.border_function, np.array([self.d_max / 2]),
                                        args=(position_i[max_index], direction[max_index, 0], self.side))['x']

            if min_index.size > 0:
                out = optimize.minimize(self.border_function, np.array([self.d_max / 2]),
                                        args=(position_i[min_index], direction[min_index, 0], 0))['x']

        elif view_index.size == 1:
            out = pairs_dist_array[view_index[0]]

        else:
            out = np.min(pairs_dist_array[view_index[0]])

        return out

    def border_function(self, x, init, direction, target):
        pos = init + x * direction
        if pos > target:
            out = pos - target
        else:
            out = target - pos
        return out

    def cost_angle_function(self, angle, desired_angle):
        pairs_angle_array, pairs_dist_array = self.angle_dist(0)
        f_angle = self.collision_function(angle, pairs_angle_array, pairs_dist_array)
        return self.d_max ** 2 + f_angle ** 2 - 2 * self.d_max * f_angle * np.cos(desired_angle - angle[0])

    def optimum_angle(self, desired_angle):
        out = optimize.minimize(self.cost_angle_function, np.array([np.pi/2]), args=desired_angle)
        return out['x']

    def angle_dist(self, i):
        point_tree = spatial.cKDTree(self.position_array)
        pairs_array = point_tree.query_pairs(10 * self.radius, output_type='ndarray')
        i_index = np.where(pairs_array == i)[0]
        pairs_array = pairs_array[i_index]
        pairs_angle_array = self.angle(pairs_array)
        possible_index = np.where(np.logical_and(pairs_angle_array < 2/3*np.pi, pairs_angle_array > 0))[0]
        pairs_angle_array = pairs_angle_array[possible_index]
        pairs_array = pairs_array[possible_index]
        pairs_dist_array = self.distance(pairs_array)
        return pairs_angle_array, pairs_dist_array

    def distance(self, pairs_array):
        array_i = self.position_array[pairs_array[:, 0]]
        array_j = self.position_array[pairs_array[:, 1]]
        return np.linalg.norm(array_i - array_j, axis=1)

    def angle(self, pairs_array):
        array_i = self.position_array[pairs_array[:, 0]]
        array_j = self.position_array[pairs_array[:, 1]]
        return np.arctan2((array_j - array_i)[:, 1], (array_j - array_i)[:, 0])

