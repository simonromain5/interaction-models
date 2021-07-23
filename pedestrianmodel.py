import numpy as np
import scipy.spatial as spatial
import basemodel as bm


class Pedestrian(bm.AbstractBwsAbpModel):
    """
    This class is an implementation of a pedestrian model greatly inspired by [1]. In addition to the model described
    in this paper, we added a target and stop mechanism: each particle has a list of possible targets (another particle
    or a particular region of the box). Each target has a given probability to be chosen. Once the target is defined,
    the particle moves until it reaches the target, stops, stays a given amount of time, then chooses a new target.
    [1]: How simple rules determine pedestrian behavior and crowd disasters.
    M Moussaïd, D Helbing, G Theraulaz - Proceedings of the National Academy of Sciences, 2011

    :param v: Speed of the particle
    :type v: float
    :param n_particles: Number of particles in the box
    :type n_particles: int
    :param angle_range: np.pi/3 by default. Range of angles where a particle 'sees' information.
    :type angle_range: float
    :param dt: 20 by default. Increment of time for each step.
    :type dt: float, optional
    :param radius: 1 by default. radius of the particles. It as constant for all the particles
    :type radius: float, optional
    :param contact_radius: distance from which we consider a contact between two particles
    :type contact_radius: float
    :param surface: 10000 by default. Surface of the box. Box is a square, hence length_side = square_root(surface)
    :type surface: float, optional
    :param n_steps: 2000 by default. Number of steps that we consider for the total movement of the particles.
    :type n_steps: int, optional
    """

    def __init__(self, v, n_particles, angle_range=np.pi / 3, dt=20, radius=1, contact_radius=2.5,
                 surface=10000, n_steps=2000):
        super().__init__(v, n_particles, dt, radius, contact_radius, surface, n_steps, False, False)
        self.angle_range = angle_range
        self.angle_array = np.arctan2(self.velocities_array[:, 1], self.velocities_array[:, 0])
        self.optimum_angle_array = np.zeros(n_particles)
        self.d_max = 10 * self.radius
        self.velocity_norm = self.v * np.ones((n_particles, 1))
        self.tau = self.d_max / self.v
        self.wall_array = np.array([[1, 0, 0], [1, 0, -self.side], [0, 1, 0], [0, 1, -self.side]])
        self.weight_array = self.initial_weight('power')

        self.target_array = np.arange(self.n_particles)
        self.target_array[self.target_array] = self.target(self.target_array)
        self.desired_position = self.position_array[self.target_array]
        self.position_array[0] = [10, 10]
        self.position_array[1] = [90, 90]
        self.previous_contact = np.arange(n_particles)
        self.real_pairs = np.empty((0, 2))
        self.pairs_iteration = np.empty((0, 3), dtype=int)

    def get_target(self):
        """
        This function returns the target of each particle

        :return: array of targets. Shape is (n_particles, 1)
        :rtype: np.array
        """
        return self.target_array

    def wall_body_collision(self, i, w, dist_iw, k=0.05):
        """
        This updates the velocity of a particle that is in contact with a wall w.

        :param i: index of particle i
        :type i: int
        :param w: wall w, which is describe by two coefficient: [a, b]. Shape is (2, )
        :type w: np.array
        :param dist_iw: distance between i and wall w
        :type dist_iw: float
        :param k: force constant
        :type k: float
        """
        force_iw = k * (self.radius - dist_iw)
        n_iw = w / np.linalg.norm(w)
        vi_t = self.velocities_array[i]
        vi_t_dt = self.dt * force_iw * n_iw / (320 * self.radius) + vi_t
        vi_norm = np.linalg.norm(vi_t_dt)

        if vi_norm != self.v:
            vi_t_dt = vi_t_dt * self.v / vi_norm
            vi_norm = self.v

        self.velocities_array[i] = vi_t_dt
        self.velocity_norm[i] = vi_norm
        self.angle_array[i] = np.arctan2(self.velocities_array[i, 1], self.velocities_array[i, 0])

    def distance_to_wall(self):
        """
        This function calculates the distance to every walls in the simulation.

        :return: distance to every wall. Shape = (n_particles, n_walls)
        :rtype: np.array
        """
        x, y = self.position_array[:, 0].reshape(-1, 1), self.position_array[:, 1].reshape(-1, 1)
        a, b, c = self.wall_array[:, 0], self.wall_array[:, 1], self.wall_array[:, 2]
        n = self.n_particles
        a, b, c = np.tile(a, (n, 1)), np.tile(b, (n, 1)), np.tile(c, (n, 1))
        numerator = np.abs(a * x + b * y + c)
        denominator = np.sqrt(a ** 2 + b ** 2)
        distance_w_array = numerator / denominator
        return distance_w_array

    def wall_collision_horizon(self, angle, i):
        """
        This function calculates the distance to collision between the particle i and every other possible wall in
        the box for a given angle. Particle i looks at all angles between -self.range and + self.range at a maximum
        distance of self.d_max. If the collision distance is superior to self.d_max, then self.d_max si returned.

        :param angle: angle particle i looks at.
        :type angle: float
        :param i: particle of index i
        :type i: int
        :return: distance to collision
        :rtype: float
        """
        a, b, c = self.wall_array[:, 0], self.wall_array[:, 1], self.wall_array[:, 2]
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
        """
        This function calculates the distance to collision between the particle i and every other possible particle in
        the box for a given angle. Particle i looks at all angles between -self.range and + self.range at a maximum
        distance of self.d_max. If the collision distance, is superior to self.d_max then self.d_max si returned.

        :param angle: angle particle i looks at.
        :type angle: float
        :param i: particle of index i
        :type i: int
        :param possible_i_index: all the particles that can be possibly in contact with i
        :type possible_i_index: np.array
        :return: distance to collision
        :rtype: float
        """
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
        dist = np.where(np.logical_and(delta == 0, t0 > 0), velocity_i_norm * t0, dist)
        dist = np.where(np.logical_and(delta > 0, t_plus > 0), velocity_i_norm * t_plus, dist)
        dist = np.where(np.logical_and(delta > 0, t_minus > 0), velocity_i_norm * t_minus, dist)
        dist = np.where(dist > self.d_max, self.d_max, dist)
        dist = np.where(dist < 0, 0, dist)
        out = np.min(dist)

        return out

    def cost_angle_function(self, angle, subj_desired_angle, i, possible_i_index, near_wall):
        """
        This function calculates the cost_angle_function for a given angle. This function is described in [1].
        Each particle i has a desired angle but has obstacle on its path. The cost function calculates the cost of a
        possible path for one angle.

        :param angle: angle that particle i looks at.
        :type angle: float
        :param subj_desired_angle: optimum angle to go to particle i 's destination if there's no obstacle.
        :type subj_desired_angle: float
        :param i: index of particle i.
        :type i: int
        :param possible_i_index: All the possible particle neighbors that can be on the path of i.
        :type possible_i_index: np.array
        :param near_wall: True if particle is near a wall, False otherwise.
        :type near_wall: bool
        :return: value of the cost function.
        :rtype: float
        """
        if possible_i_index.size > 0:
            f_angle1 = self.particle_collision_horizon(angle, i, possible_i_index)

        else:
            f_angle1 = self.d_max

        if near_wall:
            f_angle2 = self.wall_collision_horizon(angle, i)

        else:
            f_angle2 = self.d_max

        f_angle = np.min([f_angle1, f_angle2])
        return self.d_max ** 2 + f_angle ** 2 - 2 * self.d_max * f_angle * np.cos(subj_desired_angle - angle[0])

    def optimum_angle(self, subj_desired_angle, i, possible_i_index, near_wall):
        """
        This function finds the optimum angle for particle i to get to its target without entering in collision
        with other particles.

        :param subj_desired_angle: optimum angle to go to particle i 's destination if there's no obstacle.
        :type subj_desired_angle: float
        :param i: index of particle i.
        :type i: int
        :param possible_i_index: All the possible particle neighbors that can be on the path of i.
        :type possible_i_index: np.array
        :param near_wall: True if particle is near a wall, False otherwise.
        :type near_wall: bool
        :return: optimum angle that particle i should take.
        :rtype: float
        """
        possible_angle = np.linspace(-self.angle_range, self.angle_range, 750)
        y = []
        for angle in possible_angle:
            y.append(self.cost_angle_function([angle], subj_desired_angle, i, possible_i_index, near_wall))
        index_min = np.argmin(y)
        return possible_angle[index_min]

    def get_in_range_neighbors(self, free_index):
        """
        This function returns all the neighbors of all the particles in free_index.

        :param free_index: index of the particles of interest.
        :type free_index: np.array
        :return: array of neighbors.
        :rtype: np.array of lists
        """
        point_tree = spatial.cKDTree(self.position_array)
        possible_neighbors = point_tree.query_ball_point(self.position_array[free_index], self.d_max)
        in_range_neighbors = np.frompyfunc(list, 0, 1)(np.empty(possible_neighbors.size, dtype=object))

        for i, elt in enumerate(possible_neighbors):
            real_i = free_index[i]
            elt.remove(real_i)

            if len(elt) > 0:
                fictive_i_index = np.array(elt, dtype=int)
                fictive_i_position_array = self.position_array[fictive_i_index]
                fictive_i_vector_array = fictive_i_position_array - self.position_array[real_i]
                vx, vy = self.velocities_array[real_i, 0], self.velocities_array[real_i, 1]
                possible_angle_array = np.arctan2(fictive_i_vector_array[:, 1], fictive_i_vector_array[:, 0]) - \
                                       np.arctan2(vy, vx)
                possible_angle_array = np.where(possible_angle_array < - np.pi,
                                                possible_angle_array + 2 * np.pi, possible_angle_array)
                possible_angle_array = np.where(possible_angle_array > np.pi,
                                                possible_angle_array - 2 * np.pi, possible_angle_array)

                distance_array = np.linalg.norm(fictive_i_vector_array, axis=1)
                extra_angle_array = np.arcsin(2 * self.radius / distance_array)
                in_range_i_index = np.where(np.logical_and(possible_angle_array < self.angle_range + extra_angle_array,
                                                           possible_angle_array > -self.angle_range - extra_angle_array))[
                    0]
                in_range_neighbors[i].extend((fictive_i_index[in_range_i_index]))

        return in_range_neighbors

    def total_optimum_angle(self, free_index):
        """
        This function finds the optimum angle for all the particles in free_index to get to their target without
        entering in collision with other particles.
        :param free_index: index of the particles of interest.
        :type free_index: np.array
        """
        in_range_neighbors = self.get_in_range_neighbors(free_index)
        distance_to_wall_array = self.distance_to_wall()

        for i, elt in enumerate(in_range_neighbors):
            real_i = free_index[i]
            desired_vector = self.desired_position[real_i] - self.position_array[real_i]
            subj_desired_angle = np.arctan2(desired_vector[1], desired_vector[0]) - self.angle_array[real_i]
            possible_i_index = np.array(elt)
            target_i = self.target_array[i]

            if target_i in possible_i_index:
                possible_i_index = np.setdiff1d(possible_i_index, target_i)

            if possible_i_index.size > 0 and np.any(distance_to_wall_array[real_i] < self.d_max):

                optimum_angle = self.optimum_angle(subj_desired_angle, real_i, possible_i_index, True)
                distance_to_obstacle1 = self.particle_collision_horizon([optimum_angle], real_i, possible_i_index)
                distance_to_obstacle2 = self.wall_collision_horizon([optimum_angle], real_i)
                distance_to_obstacle = np.min([distance_to_obstacle1, distance_to_obstacle2])

            elif possible_i_index.size > 0:
                optimum_angle = self.optimum_angle(subj_desired_angle, real_i, possible_i_index, False)
                distance_to_obstacle = self.particle_collision_horizon([optimum_angle], real_i, possible_i_index)

            elif np.any(distance_to_wall_array[real_i] < self.d_max):
                optimum_angle = self.optimum_angle(subj_desired_angle, real_i, possible_i_index, True)
                distance_to_obstacle = self.wall_collision_horizon([optimum_angle], real_i)

            else:
                optimum_angle = subj_desired_angle
                distance_to_obstacle = self.d_max
            if np.isnan(optimum_angle):
                a=1
            self.optimum_angle_array[real_i] = optimum_angle

            if 0 < distance_to_obstacle < self.d_max:
                self.velocity_norm[real_i] = np.min([distance_to_obstacle / self.tau, self.v])

            else:
                self.velocity_norm[real_i] = self.v

    def update_velocities(self, real_contact_pairs, real_contact_index, departure_index):
        """
        This function updates the velocities of all the particles, according to what is described in [1]. The stop
        mechanism is added, and the velocities for particles in contact is set to zero. The departure mechanism is also
        added: after a contact, particles part to a new destination.

        :param real_contact_pairs: all the pairs of contact. Shape is (t, 2)
        :type real_contact_pairs: np.array
        :param real_contact_index: indexes of the particles in contact. Shape is (m, )
        :type real_contact_index: np.array
        :param departure_index: all the particles that part from their previous contact.
        :type departure_index: np.array
        """

        if real_contact_index.size > 0:
            self.velocities_array[real_contact_index] = 0
            free_index = np.setdiff1d(np.arange(self.n_particles), real_contact_index)
            free_index = np.append(free_index, departure_index)
            i_index_array, j_index_array = real_contact_pairs[:, 0], real_contact_pairs[:, 1]
            position_i_array, position_j_array = self.position_array[i_index_array], self.position_array[j_index_array]
            centers_array = position_j_array - position_i_array
            distance_array = np.linalg.norm(centers_array, axis=-1).reshape(-1, 1)
            self.position_array[i_index_array] = position_i_array - (
                    2.5 * self.radius - distance_array) * centers_array / distance_array

        else:
            free_index = np.arange(self.n_particles)

        old_velocity_norm = np.copy(self.velocity_norm[free_index])
        self.total_optimum_angle(free_index)
        cos_angle, sin_angle = np.cos(self.optimum_angle_array[free_index]), np.sin(
            self.optimum_angle_array[free_index])
        vx, vy = np.copy(self.velocities_array[free_index, 0]), np.copy(self.velocities_array[free_index, 1])
        self.velocities_array[free_index, 0] = cos_angle * vx - sin_angle * vy
        self.velocities_array[free_index, 1] = cos_angle * vy + sin_angle * vx
        zero_mask = old_velocity_norm == 0

        if np.any(zero_mask):
            self.velocities_array[zero_mask] = self.velocity_norm[zero_mask] * self.velocities_array[zero_mask]
        self.velocities_array[free_index] = self.velocities_array[free_index] / old_velocity_norm * self.velocity_norm[
            free_index]

        self.angle_array = self.optimum_angle_array + self.angle_array
        self.angle_array = np.where(self.angle_array < - np.pi, self.angle_array + 2 * np.pi, self.angle_array)
        self.angle_array = np.where(self.angle_array > np.pi, self.angle_array - 2 * np.pi, self.angle_array)

    def border(self):
        """
        This function calculates the collision of a particle with a wall.
        """
        distance_w_array = self.distance_to_wall()
        wall_coll_row, wall_coll_col = np.where(distance_w_array < self.radius)
        # add orientation and self.angle range (to do)
        for i, elt in enumerate(wall_coll_row):
            self.wall_body_collision(elt, self.wall_array[i, :2], distance_w_array[elt, wall_coll_col[i]])

    def particle_body_collision(self, i, j, dist_ij, k=5000):
        """
        This updates the velocity of particles that are in contact but aren't the target of one another.

        :param i: index of particle i
        :type i: int
        :param j: index of particle j
        :type j: int
        :param dist_ij: distance between i and j
        :type dist_ij: float
        :param k: force constant
        :type k: float
        """
        force_ij = k * (2.5 * self.radius - dist_ij)

        if force_ij == 0:
            force_ij = 1

        n_ij = (self.position_array[i] - self.position_array[j]) / dist_ij
        vi_t = self.velocities_array[i]
        vi_t_dt = self.dt * force_ij * n_ij / (320 * self.radius) + vi_t
        vi_norm = np.linalg.norm(vi_t_dt)

        if vi_norm != self.v:
            vi_t_dt = vi_t_dt * self.v / vi_norm
            vi_norm = self.v

        self.velocities_array[i] = vi_t_dt
        self.velocity_norm[i] = vi_norm
        self.angle_array[i] = np.arctan2(self.velocities_array[i, 1], self.velocities_array[i, 0])

    def particles_contact(self, false_contact_pairs):
        """
        This function updates the velocities of particles i and j that are in contact but shouldn't be.
        :param false_contact_pairs: array of all the pairs of particles that shouldn't be in contact.
        :type false_contact_pairs: np.array
        """
        for elt in false_contact_pairs:
            i, j = elt[0], elt[1]
            dist_ij = np.linalg.norm(self.position_array[j] - self.position_array[i])
            self.particle_body_collision(i, j, dist_ij)
            self.particle_body_collision(j, i, dist_ij)



    def target_probability(self, index):
        """
        This function defines at the beginning of the simulation all the types of relationship for each particle i. This
        function returns a probability array, where the elt of index (i, j) is the probability that particle i chooses
        j as a target. This array is not symmetric.
        :return: array of probabilities. Shape is (n_particles, n_particles)
        :rtype: np.array
        """
        subj_weight = np.copy(self.weight_array)
        subj_weight[index, self.target_array[index]] = 0
        norm = np.sum(subj_weight, axis=1)
        probability_array = subj_weight / norm[:, None]
        return probability_array

    def target(self, index):
        """
        This function returns a target for each particle. A target can be another particle or a specific point in the
        box.
        :param index: Index of the particles that need a new target.
        :type index: np.array
        :return: array of the new targets.
        :rtype: np.array
        """
        probability_array = self.target_probability(index)
        new_target_array = [np.random.choice(self.n_particles, p=x) for x in probability_array[index]]
        return new_target_array

    def departure(self):
        """
        This function calculates if particles in contact stay together or part from each other. The function also
        defines the new targets of particles that have just finished a contact.
        :return: Index of particles that separate
        :rtype: np.array
        """
        departure_index = []

        for elt in self.pairs_iteration:
            i = elt[0]
            j = elt[1]
            iteration = elt[2]
            prob_0 = 0.8 * (1 - np.exp(-iteration/2))
            prob_1 = 1-prob_0
            dep = np.random.choice(2, p=[prob_0, prob_1])

            if dep == 1:
                departure_index.extend([i, j])
                self.previous_contact[i] = j
                self.previous_contact[j] = i

        departure_index = np.unique(np.array(departure_index, dtype=int))

        if departure_index.size > 0:
            self.target_array[departure_index] = self.target(departure_index)

        return np.unique(np.array(departure_index))

    def initial_weight(self, output_type='uniform', k=2):
        """
        This function returns the initial weight array between all the particles that determines which will be the next
        target of a particle. This weight array is updated at each time step.
        :param output_type: 3 choices: ['uniform', 'power', 'exponential'], that will define the type of the initial weight array
        :type output_type: str
        :param k: coefficient for the 'power' and 'exponential' outputs
        :return: initial weight array
        :rtype: np.array
        """
        n = self.n_particles

        if output_type == 'uniform':
            out = np.ones((n, n))

        elif output_type == 'power':
            out = np.random.zipf(k, (n, n))

        elif output_type == 'exponential':
            out = np.random.exponential(k, (n, n))

        else:
            raise NameError(str(output_type) + ' is not in the output_type list')

        out1 = np.triu(out)
        out = out1 + np.transpose(out1)
        np.fill_diagonal(out, 0)
        return out

    def get_desired_position(self):
        """
        This function returns the desired position of each particle. When a particle targets another one, it will try
        to find the nearest point of contact with the target particle. If the particle is not attainable, then it
        follows the particle.
        :return: desired position array
        :rtype: np.array
        """
        out = np.empty((self.n_particles, 2))
        centers_array = self.position_array[self.target_array] - self.position_array
        a = np.einsum('ij,ij->i', centers_array, self.velocities_array)
        inf_mask = a <= 0
        sup_mask = ~inf_mask
        vel_norm = np.linalg.norm(self.velocities_array[self.target_array], axis=1)
        non_zero_mask = vel_norm != 0
        zero_mask = ~non_zero_mask
        non_zero_sup_mask = np.logical_and(non_zero_mask, sup_mask)
        zero_sup_mask = np.logical_and(zero_mask, sup_mask)
        vel_angle_array = np.arctan2(self.velocities_array[non_zero_sup_mask, 1],
                                     self.velocities_array[non_zero_sup_mask, 0])
        cen_angle_array = np.arctan2(centers_array[non_zero_sup_mask, 1],
                                     centers_array[non_zero_sup_mask, 0])
        beta_array = cen_angle_array - vel_angle_array
        dist_array = np.linalg.norm(centers_array[non_zero_sup_mask], axis=1)
        target_vel = self.velocities_array[self.target_array[non_zero_sup_mask]]
        normed_vel = target_vel / vel_norm[non_zero_sup_mask, None]
        dist_to_obstacle = dist_array / (2 * np.cos(beta_array))
        delta_pos = normed_vel * dist_to_obstacle[:, None]
        zero_inf_mask = np.logical_or(inf_mask, zero_sup_mask)
        out[zero_inf_mask] = self.position_array[self.target_array[zero_inf_mask]]
        out[non_zero_sup_mask] = self.position_array[self.target_array[non_zero_sup_mask]] + delta_pos
        return out

    def new_pairs_iteration(self):
        """
        This function returns the contact pairs and the number of iterations they have stayed together in one array.
        The two first columns are the pairs and the third is the number of iterations.
        :return: pairs + number of iteration the pairs have stayed together.
        :rtype: np.array.
        """
        if self.pairs_iteration.size == 0 and self.real_pairs.size == 0:
            out = np.empty((0, 3), dtype=int)

        elif self.pairs_iteration.size == 0:
            out = np.concatenate((self.real_pairs, np.ones((self.real_pairs.shape[0], 1), dtype=int)), axis=1, dtype=int)

        else:
            old_pairs = self.pairs_iteration[:, :2]
            new_in_mask = (old_pairs[:, None] == self.real_pairs).all(-1).any(-1)
            self.pairs_iteration[new_in_mask, 2] = self.pairs_iteration[new_in_mask, 2] + 1
            old_pairs_iteration = self.pairs_iteration[new_in_mask]
            old_in_mask = (self.real_pairs[:, None] == old_pairs).all(-1).any(-1)
            new_pairs = self.real_pairs[~old_in_mask]

            if new_pairs.size == 0:
                out = old_pairs_iteration

            else:
                new_pairs_iteration = np.concatenate((new_pairs, np.ones((new_pairs.shape[0], 1), dtype=int)), axis=1, dtype=int)
                out = np.concatenate((old_pairs_iteration, new_pairs_iteration), axis=0, dtype=int)

        return out

    def real_contact(self, contact_pairs):
        """
        This function finds the pairs of particles that are 'really' in contact: if two particles are in contact but not
        a target of one another, then they are not a 'real' contact. On the opposite, if two particles are in contact
        but a target of one another, then they are a 'real' contact.

        :param contact_pairs: All the pairs of contact. Shape is (m, 2)
        :type contact_pairs: np.array
        :return: index of the real pairs of contact, and the index of the particles in real contact. Shape is (2, ).
        :rtype: tuple of np.array
        """
        real_contact_index = []
        real_pairs_index = []

        for i, elt in enumerate(contact_pairs):

            if self.previous_contact[elt[0]] != elt[1]:
                real_contact_index.extend(elt)
                real_pairs_index.append(i)

        real_contact_index = np.unique(np.array(real_contact_index))
        return real_pairs_index, real_contact_index

    def iter_movement(self, step, animation=False):
        """This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds a ballistic infinitesimal step (dx, dy). Hence the new position is (x+dx, y+dy). The borders of
        the box are also considered with the self.border() function.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: This parameter is set to False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        departure_index = self.departure()
        contact_pairs, contact_index = self.contact()
        real_pairs_index, real_contact_index = self.real_contact(contact_pairs)
        self.real_pairs = contact_pairs[real_pairs_index]
        self.pairs_iteration = self.new_pairs_iteration()
        false_pairs_index = np.setdiff1d(np.arange(contact_pairs.shape[0]), real_pairs_index)
        self.update_velocities(self.real_pairs, real_contact_index, departure_index)
        self.border()
        self.particles_contact(contact_pairs[false_pairs_index])
        self.position_array = self.position_array + self.velocities_array * self.dt
        self.position_array[0] = [10, 10]
        self.position_array[1] = [90, 90]
        self.velocities_array[0] = [0.000000000000000000000001, 0.00000000000000000000000000001]
        self.velocities_array[1] = [0.00000000000000000000000001, 0.0000000000000000000000000001]

        self.desired_position = self.get_desired_position()
        self.weight_array[self.real_pairs[:, 0], self.real_pairs[:, 1]] += 1

        if not animation:
            self.creation_tij(step, contact_pairs)
