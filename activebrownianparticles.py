import numpy as np
import basemodel as bm
import scipy.spatial as spatial


class Vicsek(bm.AbstractBwsAbpModel):
    """
    This class represents the motion of particles according to the Vicsek model in 2 dimensions. The motion of the
    particles is restricted in a box and is not elastic. If stop is set to True, then the particles stop at each
    contact. In this case, they separate in the same conditions as in the BallStop class.

    :param v: Speed of the particle
    :type v: float or int
    :param n_particles: Number of particles in the box
    :type n_particles: int
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
    :param janus: False by default. Particles are janus particles if set to True.
    :type janus: bool, optional
    :param stop: stop the particle each time it encounters another one.
    :type stop: bool, optional
        """

    def __init__(self, v, n_particles, noise, dt=20, radius=1, contact_radius=2, surface=10000, n_steps=2000,
                 janus=False, stop=False):
        super().__init__(v, n_particles, dt, radius, contact_radius, surface, n_steps, janus, stop)
        self.noise = noise

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

    def update_velocities(self,  contact_pairs=None, contact_index=None):
        """
        This function updates the velocities of each particle considering the definition of the Vicsek model. Each
        particle changes its angle to align itself with is neighbors, hence the velocities vectors are updated at each
        step.
        :param contact_pairs: Default is None (if self.stop is set to False). Array of all the pairs of contact [i, j]. Shape is (n_contacts, 2).
        :type contact_pairs: np.array, optional
        :param contact_index: Default is None (if self.stop is set to False). Index of the particles in contact.
        :type contact_index: np.array
        """
        if self.stop:
            self.update_velocities_stop(contact_pairs, contact_index)
            free_index = np.delete(np.arange(self.n_particles), contact_index)

        else:
            free_index = np.arange(self.n_particles, dtype=int)

        if free_index.size > 0:
            mean_angle_array = self.mean_angle(free_index)
            noise_angle_array = (np.random.rand(free_index.size) - 0.5) * self.noise
            total_angle_array = mean_angle_array + noise_angle_array
            self.velocities_array[free_index, 0] = self.v * np.cos(total_angle_array)
            self.velocities_array[free_index, 1] = self.v * np.sin(total_angle_array)
