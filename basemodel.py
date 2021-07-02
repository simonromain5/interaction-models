import numpy as np
import scipy.spatial as spatial
import common_calc as cc


class AbstractTotalModel:

    def __init__(self, n_particles, dt, radius, surface, n_steps, janus):
        self.dt = dt
        self.radius = radius
        self.n_particles = n_particles
        self.side = np.sqrt(surface)
        self.n_steps = n_steps
        self.janus = janus
        self.tij = np.empty((0, 3))

    def get_position(self):
        """
        Returns the positions of all the particles.

        :return: An array of the positions of all the particles. It is of shape (n_particles, 2).`
        :rtype: np.array
        """
        return self.position_array

    def get_radius(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.radius

    def get_side(self):
        """
        Returns the length of the side of the box.

        :return: Length of the side of the box.`
        :rtype: float
        """
        return self.side

    def get_janus(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.janus

    def contact(self):
        """
        This function determines if one particle is in contact with other particles in the system.

        :return: A tuple of length 2. First, the neighbors array for each particle in contact and second, the respective
         index of each element in the main self.position_array.
        :rtype: tuple of np.arrays.
        """
        point_tree = spatial.cKDTree(self.position_array)
        eps = 10 ** (-3)
        contact_pairs = point_tree.query_pairs(2 * self.radius + eps, output_type='ndarray')
        contact_index = np.unique(contact_pairs)
        return contact_pairs, contact_index

    def total_movement(self):
        """
        This function iterates all the Brownian motion throughout the n_steps and returns the tij array to be analyzed

        :return: Returns the tij array. It represents all the interactions between particles i and j at time t
        :rtype: np.array
        """
        for step in range(self.n_steps):
            self.iter_movement(step)

        return np.array(self.tij)

    def creation_tij(self, step, pairs):
        """
        This function extend the tij array of all the interactions between particle i and j at time step*dt.
        This function principal role is to find the array of neighbors in a 2 * self.radius radius.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        """
        if self.janus:
            contact_i_array = pairs[:, 0]
            contact_j_array = pairs[:, 1]
            velocity_i_array, velocity_j_array = self.velocities_array[contact_i_array], \
                self.velocities_array[contact_j_array]
            truth_array = np.einsum('ij,ij->i', velocity_j_array, velocity_i_array) <= 0
            truth_index = np.where(truth_array)
            pairs = pairs[truth_index]

        time_pairs = np.append(step * self.dt * np.ones((pairs.shape[0], 1)), pairs, axis=1)
        self.tij = np.append(self.tij, time_pairs, axis=0)


class AbstractBwsAbpModel(AbstractTotalModel):

    def __init__(self, v, n_particles, dt, radius,  surface, n_steps, janus, stop):
        self.v = v
        self.stop = stop
        super().__init__(n_particles, dt, radius, surface, n_steps, janus)
        self.position_array = self.initial_positions()
        self.velocities_array = np.zeros((self.n_particles, 2))
        self.random_velocities(np.arange(0, n_particles, dtype=int))

    def get_velocities(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.velocities_array

    def get_velocities_norm(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.v

    def initial_positions(self):
        """
        This functions returns the initial position array of the particles. They are chosen randomly from an uniform
        distribution: 0 + self.radius <= x, y <= self.side - self.radius. One of the complications is that we do not
        want overlapping of the particles.

        :return: array of the initial positions of shape (self.n_particles, 2)
        :rtype: np.array
        """
        initial_positions_array = np.random.rand(self.n_particles, 2) * (self.side - 2 * self.radius) + self.radius
        point_tree = spatial.cKDTree(initial_positions_array)
        neighbors = point_tree.query_ball_point(initial_positions_array, 2 * self.radius)

        for i, elt in enumerate(neighbors):

            if len(elt) > 1:
                condition = False

                while not condition:
                    new_position = np.random.rand(2) * (self.side - 2 * self.radius) + self.radius
                    new_point_tree = spatial.cKDTree(np.delete(initial_positions_array, i, axis=0))
                    neighbors = new_point_tree.query_ball_point(new_position, 2 * self.radius + 0.1)

                    if len(neighbors) == 0:
                        initial_positions_array[i] = new_position
                        condition = True

        return initial_positions_array

    def random_velocities(self, index):
        """
        This function updates the velocities of the particles in index. The x and y components are chosen randomly but
        are subject to one constraint, the norm of all the new velocities is equal to self.v.

        :param index: index of the velocities to update
        :type index: np.array
        """
        self.velocities_array[index, 0] = (np.random.rand(index.size, 1).flatten() - 0.5) * 2 * self.v
        alg = np.random.choice([-1, 1], index.size)
        self.velocities_array[index, 1] = alg * np.sqrt(self.v ** 2 - self.velocities_array[index, 0] ** 2)

    def border(self):
        """
        This function sets to 0 the velocity of a particle that goes out of the box. The particle doesn't move out of
        the border until the randomly chosen perpendicular velocity to the border makes the particle reenter the box.
        The particle can glide on the border.
        """
        index_min = np.where(self.position_array - self.radius <= 0)
        index_max = np.where(self.position_array + self.radius >= self.side)
        self.random_velocities(index_min[0])
        self.random_velocities(index_max[0])
        index_min, column_min = np.where(np.logical_and(self.position_array - self.radius <= 0,
                                                        self.velocities_array <= 0))
        index_max, column_max = np.where(np.logical_and(self.position_array + self.radius >= self.side,
                                                        self.velocities_array >= 0))
        index_min_size = index_min.size
        index_max_size = index_max.size

        if index_min_size > 0:
            alg = np.random.choice([-1, 1], index_min_size)
            self.velocities_array[index_min, column_min] = 0
            self.velocities_array[index_min, 1-column_min] = alg * self.v

        if index_max_size > 0:
            alg = np.random.choice([-1, 1], index_max_size)
            self.velocities_array[index_max, column_max] = 0
            self.velocities_array[index_max, 1-column_max] = alg * self.v

    def update_velocities_stop(self, contact_pairs, contact_index):
        self.random_velocities(contact_index)
        i_index_array, j_index_array = contact_pairs[:, 0], contact_pairs[:, 1]
        velocity_i_array, velocity_j_array = self.velocities_array[i_index_array], self.velocities_array[
            j_index_array]
        position_i_array, position_j_array = self.position_array[i_index_array], self.position_array[j_index_array]
        centers_array = position_j_array - position_i_array
        distance_array = np.linalg.norm(centers_array, axis=-1).reshape(-1, 1)
        self.position_array[i_index_array] = position_i_array - (
                2 * self.radius - distance_array) * centers_array / distance_array
        truth_array = cc.projection(centers_array, velocity_i_array, velocity_j_array)
        velocity_pairs_null = np.where(np.logical_not(truth_array))[0]
        velocity_null_index = contact_pairs[velocity_pairs_null].ravel()
        self.velocities_array[velocity_null_index] = 0

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

        if self.stop:

            if contact_index.size > 0:
                self.update_velocities(contact_pairs, contact_index)

        else:
            self.update_velocities(None, None)

        self.border()
        self.position_array = self.position_array + self.velocities_array * self.dt

        if not animation:
            self.creation_tij(step, contact_pairs)