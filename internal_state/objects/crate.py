import numpy as np
from scipy.spatial import ConvexHull

class Crate:
    '''
    Class representing a crate which a vehicle may move
    '''
    def __init__(self, horizon, length, width, crate_id):
        self.x = np.zeros(horizon, dtype='float32')
        self.y = np.zeros(horizon, dtype='float32')
        self.theta = np.zeros(horizon, dtype='float32')
        self.length = length
        self.width = width
        self.id = crate_id
        self.vehicle = None # Vehicle currently holding crate

    def get_points(self, time):
        '''
        Returns the coordinates of the crate's corners at a given timestep
        '''
        theta = self.theta[time]
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pt1 = [self.x[time]-self.width/2., self.y[time]-self.height/2.]
        pt2 = [self.x[time]-self.width/2., self.y[time]+self.height/2.]
        pt3 = [self.x[time]+self.width/2., self.y[time]-self.height/2.]
        pt4 = [self.x[time]+self.width/2., self.y[time]+self.height/2.]
        return np.array([rot.dot(pt1), rot.dot(pt2), rot.dot(pt3), rot.dot(pt4)])

    def get_hull(self, time):
        '''
        Returns the convex hull of the points defining the crate at a given timestep
        '''
        points = self.get_points(time)
        return ConvexHull(points)

    def get_patches(self, time):
        '''
        Return the Matplotlib patches for this crate at a given timestep.
        Used for rendering purposes.
        '''
        return [Polygon(self.get_points(time))]
