import numpy as np
import scipy.spatial as spatial
import common_calc as cc


class BrownianMotion:
    """
    This class represents the Brownian motion of particles in 2 dimensions. The motion of the particles is restricted in
    a box. We consider the particles as ghost particles: they can intermingle.

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
    """

    def __init__(self, dt, radius, n_particles, surface, n_steps):
        self.n_particles = n_particles
        self.surface = surface
        self.side = np.sqrt(surface)
        self.dt = dt
        self.radius = radius
        self.n_steps = n_steps
        self.position_array = np.random.rand(self.n_particles, 2) * self.side
        self.tij = []

    def get_position(self):
        """Returns the positions of all the particles.

        :return: An array of the positions of all the particles. It is of shape (n_particles, 2)
        :rtype: np.array
        """
        return self.position_array

    def get_radius(self):
        """Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles
        :rtype: float
        """
        return self.radius

    def get_side(self):
        """Returns the length of the side of the box.

        :return: Length of the side of the box
        :rtype: float
        """
        return self.side

    def brown_iter_2d(self):
        """Returns an array of the increment of the next position of the particles (dx, dy). As we consider a Brownian
        motion, the increment follows a 2D gaussian law of mean 0 and of variance dt.

        :return: Returns an array of the increment of the next position of the particles (dx, dy). It is of shape (n_particles, 2)
        :rtype: np.array
        """
        return np.sqrt(self.dt) * np.random.randn(self.n_particles, 2)

    def iter_movement(self, step, animation=False):
        """This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds a Brownian infinitesimal step (dx, dy). Hence the new position is (x+dx, y+dy). The borders of
        the box are also considered directly in the function. If after iteration, x+dx or y+dy is inferior to 0 then the
        position is updated to x+dx=0 or y+dy=0 respectively. If after iteration, x+dx or y+dy is superior to self.side
        then the position is updated to x+dx=self.side or y+dy=self.side respectively.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: This parameter is set to False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        new_position = self.position_array + self.brown_iter_2d()
        new_position = np.where(new_position - self.radius <= 0, self.radius, new_position)
        new_position = np.where(new_position + self.radius >= self.side, self.side - self.radius, new_position)
        self.position_array = new_position
        if not animation:
            self.creation_tij(step)

    def total_movement(self):
        """
        This function iterates all the Brownian motion throughout the n_steps and returns the tij array to be analyzed

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
