import numpy as np
import scipy.spatial as spatial
import common_calc as cc


class Vicsek:
    """
        This class represents the motion of particles according to the Vicsek model in 2 dimensions. The motion of the
        particles is restricted in a box and is not elastic. If stop is set to True, then the particles stop at each
        contact. In this case, they separate in the same conditions as in the BallStop class.

        :param v: Speed of the particle
        :type v: float or int
        :param dt: Increment of time for each step. Constant * dt is the variance of the normal distribution that we use to calculate the increment of all the positions at each step.
        :type dt: float or int
        :param radius: radius of the particles. It as constant for all the particles
        :type radius: float or int
        :param n_particles: Number of particles in the box
        :type n_particles: int
        :param surface: Surface of the box. We consider the box as a square, hence the length of the side is equal to the square root of the surface.
        :type surface: float or int
        :param n_steps: Number of steps that we consider for the total movement of the particles.
        :type n_steps: int
        :param noise: Adds noise to the angle of the particle
        :type noise: float
        :param stop: stop the particle each time it encounters another one.
        :type brownian: bool, optional
        """

    def __init__(self, v, dt, radius, n_particles, surface, n_steps, noise, stop=False):
        self.v = v
        self.dt = dt
        self.radius = radius
        self.n_particles = n_particles
        self.side = np.sqrt(surface)
        self.n_steps = n_steps
        self.noise = noise
        self.stop = stop
        self.position_array = self.initial_positions()
        self.velocities_array = np.zeros((self.n_particles, 2))
        self.random_velocities(np.arange(0, n_particles, dtype=int))
        self.tij = []
        if self.stop:
            self.contact_array = np.zeros(n_particles)

    def get_position(self):
        """Returns the positions of all the particles.

        :return: An array of the positions of all the particles. It is of shape (n_particles, 2).`
        :rtype: np.array
        """
        return self.position_array

    def get_radius(self):
        """Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.radius

    def get_side(self):
        """Returns the length of the side of the box.

        :return: Length of the side of the box.`
        :rtype: float
        """
        return self.side

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

    def mean_angle(self, index):
        """
        This function calculates the mean angle of the neighbors of each particle.

        :return: Array of angles of length n_particles
        :rtype: np.array
        """
        point_tree = spatial.cKDTree(self.position_array)
        away_neighbors_array = point_tree.query_ball_point(self.position_array[index], 2.5 * self.radius)
        mean_angle_array = np.empty(index.size)
        for i, elt in enumerate(away_neighbors_array):
            v_array = np.copy(self.velocities_array[elt])
            v_array = v_array / self.v
            mean_v = np.mean(v_array, axis=0)
            mean_angle_array[i] = np.arctan2(mean_v[1], mean_v[0])
        return mean_angle_array

    def update_velocities(self,  contact_neighbors=None, contact_index=None):
        """
        This function updates the velocities of each particle considering the definition of the Vicsek model. Each
        particle changes its angle to align itself with is neighbors, hence the velocities vectors are updated at each
        step.
        """
        if self.stop:
            self.random_velocities(contact_index)
            velocity_null = []
            for i, elt in enumerate(contact_neighbors):
                real_i = contact_index[i]
                elt.remove(real_i)
                xi, yi = self.position_array[real_i]
                velocity_i = self.velocities_array[real_i]
                for j in elt:
                    xj, yj = self.position_array[j]
                    velocity_j = self.velocities_array[j]
                    part_vector = np.array([xj - xi, yj - yi])
                    if not cc.projection(part_vector, velocity_i, velocity_j):
                        if real_i not in velocity_null:
                            velocity_null.append(real_i)
            self.velocities_array[velocity_null] = 0
            free_index = np.where(self.contact_array == 0)[0]

        else:
            free_index = np.arange(0, self.n_particles, dtype=int)

        mean_angle_array = self.mean_angle(free_index)
        noise_angle_array = (np.random.rand(free_index.size) - 0.5) * self.noise
        total_angle_array = mean_angle_array + noise_angle_array
        self.velocities_array[free_index, 0] = self.v * np.cos(total_angle_array)
        self.velocities_array[free_index, 1] = self.v * np.sin(total_angle_array)

    def contact(self):
        """
        This function determines if one particle is in contact with other particles in the system.

        :return: A tuple of length 2. First, the neighbors array for each particle in contact and second, the respective
         index of each element in the main self.position_array.
        :rtype: tuple of np.arrays.
        """
        point_tree = spatial.cKDTree(self.position_array)
        neighbors = point_tree.query_ball_point(self.position_array, 2 * self.radius)
        for i, elt in enumerate(neighbors):
            if len(elt) > 1:
                self.contact_array[i] = 1
        contact_index = np.where(self.contact_array == 1)[0]
        return neighbors[contact_index], contact_index

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
        index_min = np.where(np.logical_and(self.position_array - self.radius <= 0, self.velocities_array <= 0))
        index_max = np.where(np.logical_and(self.position_array + self.radius >= self.side, self.velocities_array >= 0))
        if index_min[0].size > 0:
            for i, elt in enumerate(index_min[0]):
                column = index_min[1][i]
                alg = np.random.choice([-1, 1])
                self.velocities_array[elt, column], self.velocities_array[elt, 1-column] = 0, alg * self.v

        if index_max[0].size > 0:
            for i, elt in enumerate(index_max[0]):
                column = index_max[1][i]
                alg = np.random.choice([-1, 1])
                self.velocities_array[elt, column], self.velocities_array[elt, 1-column] = 0, alg * self.v

    def iter_movement(self, step, animation=False):
        """This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds a ballistic infinitesimal step (dx, dy). Hence the new position is (x+dx, y+dy). The borders of
        the box are also considered with the self.border() function.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: This parameter is set to False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        if self.stop:
            c_neighbors, c_index = self.contact()
            self.update_velocities(contact_neighbors=c_neighbors, contact_index=c_index)
        else:
            self.update_velocities()
        self.border()
        self.position_array += self.velocities_array * self.dt
        if self.stop:
            self.contact_array = np.zeros(self.n_particles)
        if not animation:
            self.creation_tij(step)

    def total_movement(self):
        """
        This function iterates all the ballistic motion throughout the n_steps and returns the tij array to be analyzed

        :return: Returns the tij array. It represents all the interactions between particles i and j at time t
        :rtype: np.array
        """
        for step in range(self.n_steps):
            self.iter_movement(step)
        return np.array(self.tij)

    def creation_tij(self, step):
        """
        This function extend the tij array of all the interactions between particle i and j at time step*dt.
        This function principal role is to find the array of neighbors in a 2 * self.radius radius.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        """
        point_tree = spatial.cKDTree(self.position_array)
        neighbors = point_tree.query_ball_point(self.position_array, 2 * self.radius)
        new_couples = cc.find_couples(neighbors, step * self.dt)
        self.tij.extend(new_couples)
