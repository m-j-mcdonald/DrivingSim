from matplotlib.patches import Polygon
import numpy as np

class DrivingObject(object):
    def __init__(self, x, y, theta, horizon):
        self.x = x * np.ones(horizon, dtype='float32')
        self.y = y * np.ones(horizon, dtype='float32')
        self.theta = theta * np.ones(horizon, dtype='float32')

    def get_points(self, time, dist=0):
        '''
        Returns the coordinates of the object's defining points at a given timestep.
        Must be overridden in child class.
        '''
        raise NotImplementedError

    def get_hull(self, time):
        '''
        Returns the convex hull of the points defining the object at a given timestep.
        '''
        points = self.get_points(time)
        return ConvexHull(points)

    def get_patches(self, scale, time):
        '''
        Return the Matplotlib patches for this object at a given timestep.
        Used for rendering purposes.
        '''
        return [Polygon(self.get_points(time) * scale, edgecolor='k', facecolor=self.color)]

    def update_xy_theta(self, x, y, theta, time):
        '''
        Update the x, y, theta values for a given timestep.
        '''
        old_x = self.x[time]
        old_y = self.y[time]
        old_theta = self.theta[time]
        self.x[time] = x
        self.x[time] = x
        self.theta[time] = theta

        return old_x, old_y, old_theta
