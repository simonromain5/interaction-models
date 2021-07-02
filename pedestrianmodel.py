import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize
import basemodel as bm
import matplotlib.pyplot as plt


class Pedestrian(bm.AbstractBwsAbpModel):

    def __init__(self, v, n_particles, angle_range=np.pi/3, dt=20, radius=1, surface=10000, n_steps=2000, janus=False, stop=False):
        super().__init__(v, n_particles, dt, radius, surface, n_steps, janus, stop)
        #self.velocities_array = np.array([[v, 0.001], [0, 0]])
        #self.position_array = np.array([[45, 50], [60, 50]])
        self.velocities_array = np.array([[v, 0], [-v, 0]])
        self.position_array = np.array([[30, 50], [70, 50]])
        #self.velocities_array = np.array([[v, 0], [0, -v]])
        #self.position_array = np.array([[40, 50], [50, 60]])
        self.angle_range = angle_range
        self.angle_array = np.arctan2(self.velocities_array[:, 1], self.velocities_array[:, 0])
        self.optimum_angle_array = np.zeros(n_particles)
        self.desired_angle_array = self.angle_array
        self.d_max = 10 * self.radius
        self.velocity_norm = self.v * np.ones((n_particles, 1))
        self.tau = self.d_max / self.v
        '''
        self.total_optimum_angle()
        angle_dist_list = self.angle_dist()
        elt = angle_dist_list[0]
        angle_dist_i_array = np.array(elt)
        zara = np.linspace(-np.pi/3, np.pi/3, 500)
        y=[]
        for angle in zara:
            y.append(self.collision_function([angle], angle_dist_i_array))
        plt.plot(zara, y)
        plt.show()
        a=1
        '''

    def collision_function(self, angle, angle_dist_i_array):
        angle_array, dist_array = angle_dist_i_array[:, 0], angle_dist_i_array[:, 1]
        dist_array = dist_array
        arcsin_array = np.arcsin(2 * self.radius / dist_array)

        # particle i 'sees' the particle j in the range: angle(i, j ) +- arctan(1 / d(i, j)
        view_index = np.where(np.logical_and(angle_array + arcsin_array > angle, angle_array - arcsin_array < angle))[0]
        angle_array, dist_array, arcsin_array = angle_array[view_index], dist_array[view_index], arcsin_array[view_index]


        if view_index.size == 0:
            position_i = self.position_array[0]
            direction = np.array([np.cos(angle[0]), np.sin(angle[0])])    ##REPLACE BY x/cos(theta) pus rapide
            border_vector = position_i + self.d_max * direction
            max_index = np.where(border_vector > self.side)[0]
            min_index = np.where((border_vector < 0))[0]
            out = self.d_max

            if max_index.size > 0:
                out = optimize.minimize(self.border_function, np.array([self.d_max / 2]),
                                        args=(position_i[max_index], direction[max_index], self.side))['x']

            if min_index.size > 0:
                out = optimize.minimize(self.border_function, np.array([self.d_max / 2]),
                                        args=(position_i[min_index], direction[min_index], 0))['x']

        else:
            alpha = np.abs(angle - angle_array)
            #out_array = (dist_array - self.radius * np.cos(np.pi - alpha - np.arcsin(dist_array * np.sin(alpha) / self.radius))) / np.cos(alpha)
            lim_dist = np.sqrt(np.cos(alpha) * dist_array**2 + self.radius ** 2)

            out_array = (lim_dist - dist_array) / (arcsin_array ** 2) * alpha ** 2  + dist_array

            out = np.min(out_array)
        return out -2*self.radius

    @staticmethod
    def border_function(x, init, direction, target):
        pos = init + x * direction

        if pos > target:
            out = pos - target

        else:
            out = target - pos

        return out

    def cost_angle_function(self, angle, subj_desired_angle, angle_dist_i_array):
        f_angle = self.collision_function(angle, angle_dist_i_array)
        return self.d_max ** 2 + f_angle ** 2 - 2 * self.d_max * f_angle * np.cos(subj_desired_angle - angle[0])

    def optimum_angle(self, subj_desired_angle, angle_dist_i_array):
        zara = np.linspace(-np.pi/2, np.pi/2, 500)
        y = []
        for angle in zara:
            y.append(self.cost_angle_function([angle], subj_desired_angle, angle_dist_i_array))
        '''out = optimize.minimize(self.cost_angle_function, np.array([subj_desired_angle]),
                                args=(subj_desired_angle, angle_dist_i_array))
                           '''
        index_min = np.argmin(y)
        return zara[index_min]

    def total_optimum_angle(self):
        angle_dist_list = self.angle_dist()

        for i, elt in enumerate(angle_dist_list):
            i_angle = np.arctan2(self.velocities_array[i, 1], self.velocities_array[i, 0])
            subj_desired_angle = self.desired_angle_array[i] - i_angle

            if len(elt) > 0:
                angle_dist_i_array = np.array(elt)
                opti_angle = self.optimum_angle(subj_desired_angle, angle_dist_i_array)
                self.optimum_angle_array[i] = opti_angle
                distance_to_obstacle = self.collision_function([0], angle_dist_i_array)
                if 0 < distance_to_obstacle < self.d_max - 2 * self.radius:
                    self.velocity_norm[i] = np.min([distance_to_obstacle / self.tau, self.v])

                elif distance_to_obstacle < 0:
                     self.velocity_norm[i] = 0.000000001
                else:
                    self.velocity_norm[i] = self.v

            else:
                self.optimum_angle_array[i] = subj_desired_angle
                self.velocity_norm[i] = self.v

    def angle_dist(self):
        point_tree = spatial.cKDTree(self.position_array)
        fictive_position_array = self.position_array + self.velocities_array * self.dt
        point_tree_fictive = spatial.cKDTree(fictive_position_array)
        neighbors = point_tree.query_ball_tree(point_tree_fictive, self.d_max)
        angle_dist_list = []

        for i, elt in enumerate(neighbors):
            elt.remove(i)

            if len(elt) > 0:
                fictive_i_index = np.array(elt, dtype=int)
                fictive_i_position_array = fictive_position_array[fictive_i_index]
                fictive_i_vector_array = fictive_i_position_array - self.position_array[i]
                fictive_i_vector_array = fictive_i_vector_array / np.linalg.norm(fictive_i_vector_array)
                i_index = i * np.ones(fictive_i_index.size, dtype=int)
                vel_i_array = self.velocities_array[i_index] / np.linalg.norm(self.velocities_array[i_index])
                possible_angle_array = np.arctan2(fictive_i_vector_array[:, 1], fictive_i_vector_array[:, 0]) - \
                                       np.arctan2(vel_i_array[:, 1], vel_i_array[:, 0])
                possible_angle_array = np.where(possible_angle_array < - np.pi, possible_angle_array + 2 * np.pi, possible_angle_array)
                possible_angle_array = np.where(possible_angle_array > np.pi, possible_angle_array - 2 * np.pi, possible_angle_array)
                pos_i_array = self.position_array[i_index]
                distance_array = np.linalg.norm(fictive_i_position_array - pos_i_array, axis=1)
                supplementary_angle_array = np.arcsin(2 * self.radius / distance_array)
                possible_index = np.where(np.logical_and(possible_angle_array < self.angle_range + supplementary_angle_array,
                                                         possible_angle_array > -self.angle_range - supplementary_angle_array))[0]
                possible_angle_array = possible_angle_array[possible_index]
                i_index = i * np.ones(possible_index.size, dtype=int)
                pos_i_array = self.position_array[i_index]
                pos_i_neighbors_array = fictive_i_position_array[possible_index]
                distance_array = np.linalg.norm(pos_i_neighbors_array - pos_i_array, axis=1)
                i_angle_dist_list = list(zip(list(possible_angle_array), list(distance_array)))
                angle_dist_list.append(i_angle_dist_list)

        return angle_dist_list

    def total_movement(self):
        """
        This function iterates all the Brownian motion throughout the n_steps and returns the tij array to be analyzed

        :return: Returns the tij array. It represents all the interactions between particles i and j at time t
        :rtype: np.array
        """
        for step in range(self.n_steps):
            self.iter_movement(step)

        return np.array(self.tij)

    def update_velocities(self):
        old_velocity_norm = np.copy(self.velocity_norm)
        self.total_optimum_angle()
        cos_angle, sin_angle = np.cos(self.optimum_angle_array), np.sin(self.optimum_angle_array)
        vx, vy = np.copy(self.velocities_array[:, 0]), np.copy(self.velocities_array[:, 1])
        self.velocities_array[:, 0] = cos_angle * vx - sin_angle * vy
        self.velocities_array[:, 1] = cos_angle * vy + sin_angle * vx
        zero_index = np.where(old_velocity_norm == 0)[0]
        if zero_index.size > 0:
            self.velocities_array[zero_index] = self.velocity_norm[zero_index] * self.velocities_array[zero_index]
        non_zero_index = np.setdiff1d(np.arange(self.n_particles), zero_index)
        self.velocities_array[non_zero_index] = self.velocities_array[non_zero_index] / old_velocity_norm[non_zero_index] * self.velocity_norm[non_zero_index]

        self.angle_array = self.optimum_angle_array + self.angle_array
        self.angle_array = np.where(self.angle_array < - np.pi, self.angle_array + 2 * np.pi, self.angle_array)
        self.angle_array = np.where(self.angle_array > np.pi, self.angle_array - 2 * np.pi, self.angle_array)

    def iter_movement(self, step, animation=False):
        """This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds a ballistic infinitesimal step (dx, dy). Hence the new position is (x+dx, y+dy). The borders of
        the box are also considered with the self.border() function.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: This parameter is set to False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        contact_pairs, contact_index = self.contact()
        self.update_velocities()
        self.border()
        self.position_array = self.position_array + self.velocities_array * self.dt

        if not animation:
            self.creation_tij(step, contact_pairs)
