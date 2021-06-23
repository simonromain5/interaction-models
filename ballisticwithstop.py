import numpy as np
import common_calc as cc
import basemodel as bm


class BallStop(bm.AbstractBwsAbpModel):
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

    def __init__(self, v, n_particles, dt=20, radius=1, surface=10000, n_steps=2000, brownian=False, janus=False):
        self.v = v
        self.brownian = brownian
        super().__init__(v, n_particles, dt, radius, surface, n_steps, janus, True)

    def update_velocities(self, contact_pairs, contact_index, step):
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
        i_index_array, j_index_array = contact_pairs[:, 0], contact_pairs[:, 1]
        velocity_i_array, velocity_j_array = self.velocities_array[i_index_array], self.velocities_array[j_index_array]
        position_i_array, position_j_array = self.position_array[i_index_array], self.position_array[j_index_array]
        centers_array = position_j_array - position_i_array
        distance_array = np.linalg.norm(centers_array, axis=-1).reshape(-1, 1)
        self.position_array[i_index_array] = position_i_array - (
                    2 * self.radius - distance_array) * centers_array / distance_array
        truth_array = cc.projection(centers_array, velocity_i_array, velocity_j_array)
        velocity_pairs_null = np.where(np.logical_not(truth_array))[0]
        velocity_null_index = contact_pairs[velocity_pairs_null].ravel()
        self.velocities_array[velocity_null_index] = 0

        if self.brownian:
            index = np.where(self.contact_array == 0)[0]
            dangle_array = np.sqrt(np.pi / 1000 * self.dt) * np.random.randn(index.size)
            new_velocities_array = self.velocities_array[index]
            vx, vy = np.copy(new_velocities_array[:, 0]), np.copy(new_velocities_array[:, 1])
            new_velocities_array[:, 0] = vx * np.cos(dangle_array) - vy * np.sin(dangle_array)
            new_velocities_array[:, 1] = vy * np.cos(dangle_array) + vx * np.sin(dangle_array)
            self.velocities_array[index] = new_velocities_array

    def iter_movement(self, step, animation=False):
        """This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds a ballistic infinitesimal step (dx, dy). Hence the new position is (x+dx, y+dy). The borders of
        the box are also considered with the self.border() function.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: This parameter is set to False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        contact_pairs, contact_index = self.contact(step)
        if contact_index.size > 0:
            self.update_velocities(contact_pairs, contact_index, step)
        self.border()
        self.position_array = self.position_array + self.velocities_array * self.dt
        self.contact_array = np.zeros(self.n_particles)

        if not animation:
            self.creation_tij(step, contact_pairs)


