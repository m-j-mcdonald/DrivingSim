import numpy as np
from scipy.spatial import ConvexHull

from internal_state.objects.object import DrivingObject

class Crate(DrivingObject):
    '''
    Class representing a crate which a vehicle may move
    '''
    def __init__(self, x, y, theta, horizon, length, width, crate_id):
        self.length = length
        self.width = width
        self.id = crate_id
        self.vehicle = None # Vehicle currently holding crate
        self.color = 'tan'

        super(Crate, self).__init__(x, y, theta, horizon)

    def get_points(self, time, dist=0):
        '''
        Returns the coordinates of the crate's corners at a given timestep
        '''
        theta = self.theta[time]
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        width = self.width + 2 * dist
        height = self.height + 2 * dist
        pt1 = [self.x[time]-width/2., self.y[time]-height/2.]
        pt2 = [self.x[time]-width/2., self.y[time]+height/2.]
        pt3 = [self.x[time]+width/2., self.y[time]-height/2.]
        pt4 = [self.x[time]+width/2., self.y[time]+height/2.]
        return np.array([rot.dot(pt1), rot.dot(pt2), rot.dot(pt3), rot.dot(pt4)])
