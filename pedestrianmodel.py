import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize
import basemodel as bm
import matplotlib.pyplot as plt


class Pedestrian(bm.AbstractBwsAbpModel):

    def __init__(self, v, n_particles, angle_range=np.pi/3, dt=20, radius=1, surface=10000, n_steps=2000, janus=False, stop=False):
        super().__init__(v, n_particles, dt, radius, surface, n_steps, janus, stop)
        self.velocities_array = np.array([[v, 0], [0, v]])
        self.position_array = np.array([[30, 50], [50, 30]])
        #self.velocities_array = np.array([[-v, 0], [v, 0]])
        #self.position_array = np.array([[40, 50], [60, 50]])
        #self.velocities_array = np.array([[v, 0], [0, -v]])
        #self.position_array = np.array([[40, 50], [50, 60]])
        self.angle_range = angle_range
        self.angle_array = np.arctan2(self.velocities_array[:, 1], self.velocities_array[:, 0])
        self.optimum_angle_array = np.zeros(n_particles)
        self.desired_position = np.array([[0, 50], [50, 0]])
        self.d_max = 10 * self.radius
        self.velocity_norm = self.v * np.ones((n_particles, 1))
        self.tau = self.d_max / self.v

    def wall_collision_horizon(self, angle, i):
        wall_array = np.array([[1, 0, 0], [1, 0, -self.side], [0, 1, 0], [0, 1, -self.side]])
        a, b, c = wall_array[:, 0], wall_array[:, 1], wall_array[:, 2]
        numerator1 = self.radius * np.sqrt(a ** 2 + b ** 2)
        numerator2 = a * self.position_array[i, 0] + b * self.position_array[i, 1] + c
        real_angle = self.angle_array[i] + angle[0]
        velocity_i_norm = self.velocity_norm[i]
        vx, vy = velocity_i_norm * np.cos(real_angle), velocity_i_norm * np.sin(real_angle)
        denominator = a * vx + b * vy
        denominator = np.where(denominator == 0, np.nan, denominator)

        delta_t1 = (numerator1 - numerator2) / denominator
        delta_t2 = (-numerator1 - numerator2) / denominator
        delta_t = np.stack((delta_t1, delta_t2), axis=-1)
        delta_t = np.where(delta_t < 0, self.d_max / velocity_i_norm, delta_t)

        if np.all(delta_t is np.nan):
            out = self.d_max

        else:
            delta_t_min = np.nanmin(delta_t, axis=-1)
            distance_array = velocity_i_norm * delta_t_min
            distance_array = np.where(distance_array > self.d_max, self.d_max, distance_array)
            distance_array = np.where(distance_array is np.nan, self.d_max, distance_array)
            out = np.nanmin(distance_array)

        return out

    def particle_collision_horizon(self, angle, i, possible_i_index):
        velocity_i_norm, possible_velocities = self.velocity_norm[i], self.velocities_array[possible_i_index]
        real_angle = self.angle_array[i] + angle[0]
        vx, vy = velocity_i_norm * np.cos(real_angle), velocity_i_norm * np.sin(real_angle)
        possible_velocities_x, possible_velocities_y = possible_velocities[:, 0], possible_velocities[:, 1]
        v_diff_x, v_diff_y = possible_velocities_x - vx, possible_velocities_y - vy
        position_i, possible_positions = self.position_array[i], self.position_array[possible_i_index]
        x, y = position_i[0], position_i[1]
        possible_positions_x, possible_positions_y = possible_positions[:, 0], possible_positions[:, 1]
        pos_diff_x, pos_diff_y = possible_positions_x - x, possible_positions_y - y
        a = v_diff_x ** 2 + v_diff_y ** 2
        b = 2 * (pos_diff_x * v_diff_x + pos_diff_y * v_diff_y)
        c = pos_diff_x ** 2 + pos_diff_y ** 2 - (2 * self.radius) ** 2
        delta = b ** 2 - 4 * a * c
        dist = self.d_max * np.ones(delta.size)
        delta = np.where(delta < 0, np.nan, delta)
        b_prime = np.where(b == 0, np.nan, b)
        t_a = - c / b_prime
        dist = np.where(np.logical_and(a == 0, t_a > 0), velocity_i_norm * t_a, dist)
        a = np.where(a == 0, np.nan, a)
        t0 = - b / (2 * a)

        t_plus, t_minus = (- b + np.sqrt(delta)) / (2 * a), (- b - np.sqrt(delta)) / (2 * a)
        dist = np.where(np.logical_and(delta == 0, t0 > 0),  velocity_i_norm * t0, dist)
        dist = np.where(np.logical_and(delta > 0, t_plus > 0), velocity_i_norm * t_plus, dist)
        dist = np.where(np.logical_and(delta > 0, t_minus > 0), velocity_i_norm * t_minus, dist)
        dist = np.where(dist > self.d_max, self.d_max, dist)
        dist = np.where(dist < 0, 0, dist)
        out = np.min(dist)

        return out

    def cost_angle_function(self, angle, subj_desired_angle, i, possible_i_index):

        if possible_i_index.size > 0:
            f_angle1 = self.particle_collision_horizon(angle, i, possible_i_index)

        else:
            f_angle1 = self.d_max

        f_angle2 = self.wall_collision_horizon(angle, i)
        f_angle = np.min([f_angle1, f_angle2])
        if np.abs(angle[0]) < 0.003:
            a=1
        if f_angle < self.d_max-6:
            a=1
        return self.d_max ** 2 + f_angle ** 2 - 2 * self.d_max * f_angle * np.cos(subj_desired_angle - angle[0])

    def optimum_angle(self, subj_desired_angle, i, possible_i_index):
        zara = np.linspace(-self.angle_range, self.angle_range, 500)
        y = []
        for angle in zara:
            y.append(self.cost_angle_function([angle], subj_desired_angle, i, possible_i_index))
        '''out = optimize.minimize(self.cost_angle_function, np.array([subj_desired_angle]),
                                args=(subj_desired_angle, angle_dist_i_array))
                           '''
        index_min = np.argmin(y)
        return zara[index_min]

    def total_optimum_angle(self):
        in_range_neighbors = self.get_in_range_neighbors()

        for i, elt in enumerate(in_range_neighbors):
            desired_vector = self.desired_position[i] - self.position_array[i]
            subj_desired_angle = np.arctan2(desired_vector[1], desired_vector[0]) - self.angle_array[i]
            possible_i_index = np.array(elt)
            optimum_angle = self.optimum_angle(subj_desired_angle, i, possible_i_index)
            self.optimum_angle_array[i] = optimum_angle

            if possible_i_index.size > 0:
                distance_to_obstacle1 = self.particle_collision_horizon([optimum_angle], i, possible_i_index)

            else:
                distance_to_obstacle1 = self.d_max

            distance_to_obstacle2 = self.wall_collision_horizon([optimum_angle], i)
            distance_to_obstacle = np.min([distance_to_obstacle1, distance_to_obstacle2])

            if 0 < distance_to_obstacle < self.d_max:
                self.velocity_norm[i] = np.min([distance_to_obstacle / self.tau, self.v])

            else:
                self.velocity_norm[i] = self.v

    def get_in_range_neighbors(self):
        point_tree = spatial.cKDTree(self.position_array)
        possible_neighbors = point_tree.query_ball_point(self.position_array, self.d_max)
        in_range_neighbors = np.frompyfunc(list, 0, 1)(np.empty(possible_neighbors.size, dtype=object))

        for i, elt in enumerate(possible_neighbors):
            elt.remove(i)

            if len(elt) > 0:
                fictive_i_index = np.array(elt, dtype=int)
                fictive_i_position_array = self.position_array[fictive_i_index]
                fictive_i_vector_array = fictive_i_position_array - self.position_array[i]
                vx, vy = self.velocities_array[i, 0], self.velocities_array[i, 1]
                possible_angle_array = np.arctan2(fictive_i_vector_array[:, 1], fictive_i_vector_array[:, 0]) - \
                    np.arctan2(vy, vx)
                possible_angle_array = np.where(possible_angle_array < - np.pi,
                                                possible_angle_array + 2 * np.pi, possible_angle_array)
                possible_angle_array = np.where(possible_angle_array > np.pi,
                                                possible_angle_array - 2 * np.pi, possible_angle_array)

                distance_array = np.linalg.norm(fictive_i_vector_array, axis=1)
                extra_angle_array = np.arcsin(2 * self.radius / distance_array)
                in_range_i_index = np.where(np.logical_and(possible_angle_array < self.angle_range + extra_angle_array,
                                            possible_angle_array > -self.angle_range - extra_angle_array))[0]
                in_range_neighbors[i].extend(fictive_i_index[in_range_i_index])

        return in_range_neighbors

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

    def total_movement(self):
        """
        This function iterates all the Brownian motion throughout the n_steps and returns the tij array to be analyzed

        :return: Returns the tij array. It represents all the interactions between particles i and j at time t
        :rtype: np.array
        """
        for step in range(self.n_steps):
            self.iter_movement(step)

        return np.array(self.tij)