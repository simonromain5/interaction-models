import numpy as np
import scipy.spatial as spatial
import common_calc as cc


class BallStop:
    """
    This class represents the ballistic motion of particles in 2 dimensions with a stop at each contact. The motion of
    the particles is restricted in a box and is not elastic. When two particles are in contact, their new velocity
    vectors are taken randomly. One of the option of the class is to set brownian to True. In this case, the particle
    will ally ballistic motion and a brownian motion.

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
    :param brownian: Adds a brownian motion to the particles if set to True
    :type brownian: bool, optional
    """

    def __init__(self, v, dt, radius, n_particles, surface, n_steps, brownian=False, janus=False):
        self.v = v
        self.dt = dt
        self.radius = radius
        self.n_particles = n_particles
        self.side = np.sqrt(surface)
        self.n_steps = n_steps
        self.brownian = brownian
        self.janus = janus
        self.position_array = self.initial_positions()
        self.velocities_array = np.zeros((n_particles, 2))
        self.contact_array = np.zeros(n_particles)
        self.random_velocities(np.arange(0, n_particles, dtype=int))
        self.tij = []

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

    def contact(self):
        """
        This function determines if one particle is in contact with other particles in the system.

        :return: A tuple of length 2. First, the neighbors array for each particle in contact and second, the respective
         index of each element in the main self.position_array.
        :rtype: tuple of np.arrays.
        """
        point_tree = spatial.cKDTree(self.position_array)
        neighbors = point_tree.query_ball_point(self.position_array, 2 * self.radius)

        if self.janus:
            non_contact_array = []
            new_speed_array = []

            for i, elt in enumerate(neighbors):
                condition = False

                if len(elt) > 1:
                    vi = self.velocities_array[i]

                    if not np.any(vi):
                        condition = True

                    else:

                        for j in elt:
                            if i == j:
                                continue

                            vj = self.velocities_array[j]

                            if np.dot(vi, vj) <= 0:
                                condition = True
                                break

                            else:
                                v = vj
                                v_per = np.array([-vj[1], vj[0]])

                    if condition:
                        self.contact_array[i] = 1

                    else:
                        non_contact_array.append(i)
                        cos = np.dot(v, vi) / self.v**2
                        sin = np.cross(v, vi) / self.v**2
                        new_speed_array.append(cos * v - sin * v_per)

            if len(non_contact_array) > 0:
                non_contact_array = np.array(non_contact_array, dtype=int)
                self.velocities_array[non_contact_array] = np.array(new_speed_array)

        else:

            for i, elt in enumerate(neighbors):

                if len(elt) > 1:
                    self.contact_array[i] = 1

        contact_index = np.where(self.contact_array == 1)[0]
        return neighbors[contact_index], contact_index

    def update_velocities(self, contact_neighbors, contact_index):
        """
        This function updates the velocities of all the particles. For all the particles that are not in contact with
        another particle, then their velocities are not updated, unless self.brownian is set to True (The particles will
        have a brownian motion). For all the particles that are in contact with another particle, then their velocities
        are updated randomly. Consider two particles i and j, and the vector u that link the center of i with the center
        of j. If the new velocities vi and vj are such as u.vi <= 0 <= u.vj, then the particles keep these velocities.
        Else, vi and vj are set to 0. The probability that two particles part from each other is 1/4.

        :param contact_neighbors: Contact_neighbors[i] lists all the neighbors of particle contact_index[i]
        :type contact_neighbors: np.array of list
        :param contact_index: index of the real indexes of the particles in contact
        :type contact_index: np.array

        """
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

        if self.brownian:
            index = np.where(self.contact_array == 0)[0]
            dangle_array = np.sqrt(np.pi / 1000 * self.dt) * np.random.randn(index.size)
            new_velocities_array = self.velocities_array[index]
            vx, vy = np.copy(new_velocities_array[:, 0]), np.copy(new_velocities_array[:, 1])
            new_velocities_array[:, 0] = vx * np.cos(dangle_array) - vy * np.sin(dangle_array)
            new_velocities_array[:, 1] = vy * np.cos(dangle_array) + vx * np.sin(dangle_array)
            self.velocities_array[index] = new_velocities_array

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
        contact_neighbors, contact_index = self.contact()
        self.update_velocities(contact_neighbors, contact_index)
        self.border()
        self.position_array = self.position_array + self.velocities_array * self.dt
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

        if not self.janus:
            new_couples = cc.find_couples(neighbors, step * self.dt)

        else:
            new_couples = cc.find_couples(neighbors, step * self.dt, self.velocities_array, janus=True)

        self.tij.extend(new_couples)

