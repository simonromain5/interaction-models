import numpy as np
import common_calc as cc
import basemodel as bm
import scipy.spatial as spatial


class Vicsek(bm.AbstractBwsAbpModel):
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

    def __init__(self, v, n_particles, noise, dt=20, radius=1, surface=10000, n_steps=2000, janus=False, stop=False):
        super().__init__(v, n_particles, dt, radius, surface, n_steps, janus, stop)
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
                position_i = self.position_array[real_i]
                velocity_i = self.velocities_array[real_i]
                neighbors_i = np.array(elt, dtype=int)
                neighbors_i_position = self.position_array[neighbors_i]
                neighbors_i_velocity = self.velocities_array[neighbors_i]
                centers_array = neighbors_i_position - position_i
                if not cc.projection(centers_array, velocity_i, neighbors_i_velocity):
                    velocity_null.append(real_i)

            self.velocities_array[velocity_null] = 0
            free_index = np.where(self.contact_array == 0)[0]

        else:
            free_index = np.arange(0, self.n_particles, dtype=int)

        if free_index.size > 0:
            mean_angle_array = self.mean_angle(free_index)
            noise_angle_array = (np.random.rand(free_index.size) - 0.5) * self.noise
            total_angle_array = mean_angle_array + noise_angle_array
            self.velocities_array[free_index, 0] = self.v * np.cos(total_angle_array)
            self.velocities_array[free_index, 1] = self.v * np.sin(total_angle_array)

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
            c_neighbors, c_index = self.contact(step)
            self.update_velocities(contact_neighbors=c_neighbors, contact_index=c_index)

        else:
            self.update_velocities()

        self.border()
        self.position_array += self.velocities_array * self.dt

        if self.stop:
            self.contact_array = np.zeros(self.n_particles)

        if not animation:
            self.creation_tij(step)
